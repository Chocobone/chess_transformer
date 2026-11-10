"""학습 폴더의 어댑터로 val 전체를 다시 평가하고, 그 폴더 log.jsonl의 마지막 eval 줄과 비교한다.

받은 어댑터 파일이 학습 끝의 가중치 그대로인지, 다른 장비에서도 같은 값이 나오는지 확인한다.
평가 조건(데이터, 손실 설정, 묶음 크기, 정밀도)은 폴더의 config.json에서 읽어 학습 때와 같게 맞춘다.

    python -m gpu_train.eval_adapter runs/gpu/R1 --baseline --json runs/gpu/R1/reeval.json

판정 (서버 smoke와 같은 규칙): top1·top3는 정확히 같고, 나머지 지표는 절대 차이가 5 × 10⁻⁵ 미만이다.
모두 통과하면 종료 코드 0, 하나라도 실패하면 1이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PositionDataset, collate_fn, player_dir
from src.model.lora import load_adapter
from src.model.transformer import load_base
from src.train.train import Objective, evaluate

METRICS = ("imitation", "top1", "top3", "regret", "regret_top1", "band_violation", "strength")
EXACT = ("top1", "top3")
TOLERANCE = 5e-5


def read_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def read_reference(run_dir: Path) -> dict[str, dict]:
    """log.jsonl의 baseline 줄과 마지막 eval 줄."""
    records = [json.loads(line) for line in open(run_dir / "log.jsonl", encoding="utf-8") if line.strip()]
    baselines = [r for r in records if r["type"] == "baseline"]
    evals = [r for r in records if r["type"] == "eval"]
    if not evals:
        raise ValueError(f"{run_dir / 'log.jsonl'}에 eval 줄이 없다")
    return {"baseline": baselines[0] if baselines else None, "eval": evals[-1]}


def compare(metrics: dict[str, float], reference: dict, tolerance: float = TOLERANCE) -> dict:
    """지표별 차이(다시 잰 값 − 기록값)와 판정. top1·top3는 정확히 같아야 하고, 나머지는 |차이| < tolerance."""
    items = {}
    for name in METRICS:
        diff = metrics[name] - reference[name]
        passed = diff == 0 if name in EXACT else abs(diff) < tolerance
        items[name] = {"value": metrics[name], "reference": reference[name], "diff": diff, "pass": passed}
    return {"pass": all(item["pass"] for item in items.values()), "items": items}


def val_loader(config: dict, num_workers: int, base: str, data_dir: str | None) -> DataLoader:
    """train.py와 같은 val 로더: 앞쪽 eval_positions개(0이면 전체), 섞지 않음, batch_positions씩."""
    dataset = PositionDataset(Path(data_dir) if data_dir else player_dir(config["player"]), "val", base)
    if config.get("eval_positions"):
        dataset = Subset(dataset, range(min(config["eval_positions"], len(dataset))))
    if len(dataset) != config["val_positions"]:
        raise ValueError(f"val 국면 수가 학습 때와 다르다: {len(dataset)} != {config['val_positions']}")
    return DataLoader(dataset, batch_size=config["batch_positions"], shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reevaluate(run_dir: Path, device: str, num_workers: int = 2, baseline: bool = False,
               base: str | None = None, data_dir: str | None = None) -> dict:
    """폴더의 어댑터를 다시 평가하고 기록과 비교한 결과."""
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    reference = read_reference(run_dir)
    base = base or config["base"]
    data_dir = data_dir or config.get("data_dir")
    device = torch.device(device)
    objective = Objective(config["temperature"], config["tau"], config["lam"], config["alpha"], config["gamma"])
    # 학습 중 평가와 같은 정밀도. 기본은 float32이고, bfloat16은 학습 때 eval_amp를 켠 경우에만 쓴다
    amp = bool(config["amp"] and config["eval_amp"] and device.type == "cuda")
    loader = val_loader(config, num_workers, base, data_dir)

    result = {
        "run_dir": str(run_dir),
        "adapter_sha256": sha256(run_dir / "adapter.pt"),
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "settings": {"base": str(base), "data_dir": str(data_dir), "val_positions": len(loader.dataset),
                     "eval_max_moves": config["eval_max_moves"], "eval_amp": amp},
    }

    start = time.perf_counter()
    model = load_base(base, device)
    result["adapter_settings"] = load_adapter(model, run_dir / "adapter.pt")
    model.to(device).eval()
    metrics = evaluate(model, loader, objective, config["eval_max_moves"], amp)
    result["eval"] = {"step": reference["eval"]["step"], **compare(metrics, reference["eval"])}
    result["seconds"] = {"eval": time.perf_counter() - start}

    if baseline:
        if reference["baseline"] is None:
            raise ValueError(f"{run_dir / 'log.jsonl'}에 baseline 줄이 없다 (재개만 한 기록)")
        start = time.perf_counter()
        metrics = evaluate(None, loader, objective, config["eval_max_moves"], device=device)
        result["baseline"] = compare(metrics, reference["baseline"])
        result["seconds"]["baseline"] = time.perf_counter() - start

    result["pass"] = result["eval"]["pass"] and result.get("baseline", {"pass": True})["pass"]
    return result


def report(result: dict) -> str:
    lines = [f"폴더 {result['run_dir']}",
             f"adapter.pt SHA-256 {result['adapter_sha256']}",
             "환경 " + ", ".join(f"{k} {v}" for k, v in result["environment"].items()),
             f"val {result['settings']['val_positions']}국면, eval_max_moves {result['settings']['eval_max_moves']}, "
             f"bfloat16 {result['settings']['eval_amp']}"]
    for part in ("eval", "baseline"):
        if part not in result:
            continue
        title = f"마지막 eval (step {result['eval']['step']})" if part == "eval" else "baseline"
        lines.append(f"== {title}: {'통과' if result[part]['pass'] else '실패'} "
                     f"({result['seconds'][part]:.1f}초)")
        for name, item in result[part]["items"].items():
            mark = "OK " if item["pass"] else "NG "
            lines.append(f"  {mark}{name:<15}{item['value']:.10f}  기록 {item['reference']:.10f}  "
                         f"차이 {item['diff']:+.2e}")
    lines.append(f"판정: {'통과' if result['pass'] else '실패'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="학습 폴더의 어댑터를 다시 평가해 log.jsonl의 마지막 eval과 비교한다")
    parser.add_argument("run_dir", type=Path, help="config.json, log.jsonl, adapter.pt가 있는 학습 폴더")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--baseline", action="store_true", help="베이스 기준선도 다시 재서 baseline 줄과 비교한다")
    parser.add_argument("--base", help="베이스 가중치 경로 (기본: config.json의 base)")
    parser.add_argument("--data-dir", help="데이터 폴더 (기본: config.json의 data_dir, 없으면 기사 기본 폴더)")
    parser.add_argument("--json", type=Path, help="결과를 이 경로에 JSON으로 남긴다")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reevaluate(args.run_dir, args.device, args.num_workers, args.baseline, args.base, args.data_dir)
    print(report(result), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
