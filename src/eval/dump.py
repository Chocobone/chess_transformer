"""학습 폴더의 어댑터를 세 기사 val 전체(S_val)에서 평가해 합법수별 log π를 파일로 남긴다.

    python -m src.eval.dump runs/gpu/R1 --out runs/gpu/dump

`{out}`에 남는 파일 (세 기사 val을 `PLAYERS` 순서로 이어 붙인다. 국면 N개, 합법수 M개):

    positions.npz   모델과 무관한 국면 정보. 처음 덤프할 때 한 번 만든다
                    player[N] game[N] ply[N] offsets[N+1] target[N] regret[M] new[N], settings(JSON 문자열)
    base.npz        log_pi[M]: 캐시의 베이스 q로 만든 정책 (학습의 baseline과 같은 계산)
    {모델 ID}.npz   log_pi[M]: 어댑터의 정책. 모델 ID는 학습 폴더 이름이다
    {모델 ID}.json  학습 기록 대조, 환경, 어댑터 SHA-256, 걸린 시간

정책과 평가 묶음은 학습 중 평가와 같다: 학습 config.json의 temperature, batch_positions(섞지 않음),
eval_max_moves, 정밀도(eval_amp가 꺼져 있으면 float32). 그래서 자기 기사 val에서는 덤프로 다시 계산한
지표가 log.jsonl의 마지막 eval과 같아야 한다 (규칙은 gpu_train/eval_adapter.py와 같다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PositionDataset, collate_fn, player_dir, read_records, split_by_moves
from src.eval.metrics import compare, count_ties, per_position, summarize
from src.model.lora import load_adapter
from src.model.transformer import load_base
from src.players import PLAYERS
from src.train.policy import policy_from_q

PLAYER_ORDER = tuple(PLAYERS)


def fen_key(fen: str) -> str:
    """배치·차례·캐슬링·앙파상 (FEN 앞 4필드). Task #6 겹침 계산과 같은 키다."""
    return " ".join(fen.split()[:4])


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_of(config: dict) -> dict:
    return {
        "base": Path(config["base"]).name,
        "temperature": config["temperature"],
        "tau": config["tau"],
        "batch_positions": config["batch_positions"],
        "eval_max_moves": config["eval_max_moves"],
        "eval_amp": bool(config["amp"] and config["eval_amp"]),
    }


def data_dir_of(player: str, data_root: Path | None) -> Path:
    return Path(data_root) / player if data_root else player_dir(player)


def val_loader(player: str, base: str, settings: dict, data_root: Path | None, num_workers: int) -> DataLoader:
    dataset = PositionDataset(data_dir_of(player, data_root), "val", base)
    return DataLoader(dataset, batch_size=settings["batch_positions"], shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers)


@torch.no_grad()
def policy_log_pi(model, loader: DataLoader, settings: dict, device: torch.device) -> np.ndarray:
    """합법수별 log π를 국면 순서, 국면 안 행동 번호 순으로 편 배열. model이 None이면 캐시의 베이스 q를 쓴다."""
    amp = bool(settings["eval_amp"] and device.type == "cuda")
    parts = []
    for batch in loader:
        for chunk in split_by_moves(batch, settings["eval_max_moves"]):
            chunk = chunk.to(device)
            if model is None:
                q = chunk.q_base
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    q = model.win_prob(chunk.tokens)
            log_pi, mask = policy_from_q(q, chunk.owner, settings["temperature"])
            parts.append(log_pi[mask].float().cpu().numpy())  # 행 우선 순서 = 국면 순, 국면 안 수 순
    return np.concatenate(parts)


def build_positions(base: str, settings: dict, data_root: Path | None) -> dict[str, np.ndarray]:
    """세 기사 val의 국면 정보. 캐시의 합법수 순서와 베이스 q를 그대로 쓴다."""
    train_keys = set()
    for player in PLAYER_ORDER:
        train_keys.update(fen_key(record["fen"]) for record in read_records(data_dir_of(player, data_root), "train"))

    columns = {name: [] for name in ("player", "game", "ply", "target", "new", "counts", "regret")}
    for index, player in enumerate(PLAYER_ORDER):
        data_dir = data_dir_of(player, data_root)
        records = read_records(data_dir, "val")
        dataset = PositionDataset(data_dir, "val", base)
        for i, record in enumerate(records):
            span = slice(dataset.offsets[i], dataset.offsets[i + 1])
            actions, q = dataset.actions[span], dataset.q[span]
            (hit,) = np.nonzero(actions == dataset.targets[i])
            if len(hit) != 1:
                raise ValueError(f"{player} val {i}: 기사 수가 합법수 목록에 없다")
            columns["player"].append(index)
            columns["game"].append(record["game_id"])
            columns["ply"].append(record["ply"])
            columns["target"].append(int(hit[0]))
            columns["new"].append(record["ply"] >= 20 and fen_key(record["fen"]) not in train_keys)
            columns["counts"].append(len(actions))
            columns["regret"].append(q.max() - q)  # 학습 평가와 같은 float32 계산
    return {
        "player": np.array(columns["player"], dtype=np.int8),
        "game": np.array(columns["game"], dtype=np.int32),
        "ply": np.array(columns["ply"], dtype=np.int16),
        "offsets": np.concatenate([[0], np.cumsum(columns["counts"])]).astype(np.int64),
        "target": np.array(columns["target"], dtype=np.int16),
        "regret": np.concatenate(columns["regret"]).astype(np.float32),
        "new": np.array(columns["new"], dtype=bool),
    }


def load_positions(out: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(Path(out) / "positions.npz") as data:
        arrays = {name: data[name] for name in data.files if name != "settings"}
        settings = json.loads(str(data["settings"]))
    return arrays, settings


def ensure_positions(out: Path, base: str, settings: dict, data_root: Path | None, num_workers: int) -> dict:
    """positions.npz와 base.npz를 만든다. 이미 있으면 설정이 같은지만 확인한다."""
    out = Path(out)
    if (out / "positions.npz").exists() and (out / "base.npz").exists():
        _, saved = load_positions(out)
        if saved != settings:
            raise ValueError(f"{out}의 덤프 설정이 이 모델과 다르다: {saved} != {settings}")
        return saved
    out.mkdir(parents=True, exist_ok=True)
    positions = build_positions(base, settings, data_root)
    log_pi = np.concatenate([
        policy_log_pi(None, val_loader(player, base, settings, data_root, num_workers), settings, torch.device("cpu"))
        for player in PLAYER_ORDER
    ])
    if len(log_pi) != positions["offsets"][-1]:
        raise ValueError(f"베이스 정책의 합법수 수({len(log_pi)})가 캐시({positions['offsets'][-1]})와 다르다")
    np.savez(out / "base.npz", log_pi=log_pi)
    np.savez(out / "positions.npz", settings=json.dumps(settings, sort_keys=True), **positions)
    return settings


def read_reference(run_dir: Path) -> dict:
    records = [json.loads(line) for line in open(Path(run_dir) / "log.jsonl", encoding="utf-8") if line.strip()]
    evals = [r for r in records if r["type"] == "eval"]
    baselines = [r for r in records if r["type"] == "baseline"]
    if not evals or not baselines:
        raise ValueError(f"{run_dir / 'log.jsonl'}에 eval 또는 baseline 줄이 없다")
    return {"eval": evals[-1], "baseline": baselines[0]}


def check_against_log(out: Path, model_id: str, player: str, settings: dict, reference: dict) -> dict:
    """덤프 파일에서 자기 기사 val의 7개 지표를 다시 계산해 학습 기록과 비교한다."""
    positions, _ = load_positions(out)
    select = positions["player"] == PLAYER_ORDER.index(player)
    first, last = np.nonzero(select)[0][[0, -1]]
    offsets = positions["offsets"][first:last + 2]
    span = slice(int(offsets[0]), int(offsets[-1]))
    local = offsets - offsets[0]
    result = {}
    for part, name in (("eval", model_id), ("baseline", "base")):
        with np.load(Path(out) / f"{name}.npz") as data:
            log_pi = data["log_pi"][span]
        values = per_position(log_pi, local, positions["target"][select], positions["regret"][span], settings["tau"])
        result[part] = compare(summarize(values), reference[part])
        result[part]["tied_positions"] = int(np.sum(count_ties(log_pi, local) > 1))
    result["eval"]["step"] = reference["eval"]["step"]
    return result


def dump_model(run_dir: Path, out: Path, device: str, num_workers: int = 2, data_root: Path | None = None,
               base: str | None = None) -> dict:
    """학습 폴더 하나를 덤프하고 학습 기록과 대조한다. 이미 통과한 덤프가 있으면 건너뛴다."""
    run_dir, out = Path(run_dir), Path(out)
    model_id = run_dir.name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    base = base or config["base"]
    settings = settings_of(config)
    adapter_sha = sha256(run_dir / "adapter.pt")
    record_path = out / f"{model_id}.json"
    if record_path.exists() and (out / f"{model_id}.npz").exists():
        saved = json.loads(record_path.read_text(encoding="utf-8"))
        if saved.get("pass") and saved.get("adapter_sha256") == adapter_sha:
            saved["skipped"] = True
            return saved

    device = torch.device(device)
    start = time.perf_counter()
    ensure_positions(out, base, settings, data_root, num_workers)
    seconds = {"positions": time.perf_counter() - start}

    model = load_base(base, device)
    adapter_settings = load_adapter(model, run_dir / "adapter.pt")
    model.to(device).eval()
    log_pi, seconds["players"] = [], {}
    for player in PLAYER_ORDER:
        start = time.perf_counter()
        log_pi.append(policy_log_pi(model, val_loader(player, base, settings, data_root, num_workers), settings, device))
        seconds["players"][player] = time.perf_counter() - start
    log_pi = np.concatenate(log_pi)
    positions, _ = load_positions(out)
    if len(log_pi) != positions["offsets"][-1]:
        raise ValueError(f"{model_id}의 합법수 수({len(log_pi)})가 positions.npz({positions['offsets'][-1]})와 다르다")
    np.savez(out / f"{model_id}.npz", log_pi=log_pi)

    result = {
        "model": model_id,
        "run_dir": str(run_dir),
        "player": config["player"],
        "adapter_sha256": adapter_sha,
        "adapter_settings": adapter_settings,
        "settings": settings,
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "positions": int(len(positions["target"])),
        "moves": int(positions["offsets"][-1]),
        "seconds": seconds,
    }
    result.update(check_against_log(out, model_id, config["player"], settings, read_reference(run_dir)))
    result["pass"] = result["eval"]["pass"] and result["baseline"]["pass"]
    with open(record_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def report(result: dict) -> str:
    if result.get("skipped"):
        return f"{result['model']}: 이미 덤프했고 대조 통과 (건너뜀)"
    lines = [f"모델 {result['model']} ({result['player']}), 폴더 {result['run_dir']}",
             f"adapter.pt SHA-256 {result['adapter_sha256']}",
             "환경 " + ", ".join(f"{k} {v}" for k, v in result["environment"].items()),
             f"S_val {result['positions']}국면, 합법수 {result['moves']}개, 설정 {result['settings']}",
             "시간 " + ", ".join(f"{p} {s:.1f}초" for p, s in result["seconds"]["players"].items())]
    for part in ("eval", "baseline"):
        title = f"마지막 eval (step {result['eval']['step']})" if part == "eval" else "baseline (base.npz)"
        lines.append(f"== {title}: {'통과' if result[part]['pass'] else '실패'} "
                     f"(1순위 동률 국면 {result[part]['tied_positions']}개)")
        for name, item in result[part]["items"].items():
            mark = "OK " if item["pass"] else "NG "
            lines.append(f"  {mark}{name:<15}{item['value']:.10f}  기록 {item['reference']:.10f}  "
                         f"차이 {item['diff']:+.2e}")
    lines.append(f"판정: {'통과' if result['pass'] else '실패'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="학습 폴더의 어댑터를 S_val에서 평가해 합법수별 log π를 남긴다")
    parser.add_argument("run_dir", type=Path, help="config.json, log.jsonl, adapter.pt가 있는 학습 폴더")
    parser.add_argument("--out", type=Path, default=Path("runs/gpu/dump"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, help="기사별 데이터 폴더의 상위 폴더 (기본 data/processed)")
    parser.add_argument("--base", help="베이스 가중치 경로 (기본: config.json의 base)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """종료 코드: 0 대조 통과, 1 대조 실패, 3 덤프 중 오류 (dump.sbatch가 실패와 오류를 구분한다)."""
    args = build_parser().parse_args(argv)
    try:
        result = dump_model(args.run_dir, args.out, args.device, args.num_workers, args.data_root, args.base)
    except Exception:
        traceback.print_exc()
        return 3
    print(report(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
