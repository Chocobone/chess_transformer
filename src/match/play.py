"""대국 실행 CLI (Task #13). 평가 대상 모델 하나와 상대 하나로 짝 대국을 둔다.

    python -m src.match.play --model R1 --opponent base --openings 0-199 --out runs/gpu/match/C1_R1
    python -m src.match.play --model base --opponent sf --nodes 10000 --openings 0-49 --out runs/gpu/match/cal_base_n10000

- 모델 ID "base"는 베이스 136M, 그 밖의 ID는 `{--runs-dir}/{ID}/adapter.pt`다
- 같은 명령을 다시 실행하면 끝난 대국을 건너뛰고 이어서 둔다. 설정(모델, 상대, 노드 수)이 기존 기록과 다르면 거부한다
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from src.match.game import pair_specs
from src.match.openings import DEFAULT_OUT as OPENINGS_PATH
from src.match.openings import load_openings, parse_range
from src.match.policy import BASE_ID, PolicyBank
from src.match.runner import SF_ID, StockfishPool, run


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def check_config(out: Path, config: dict) -> None:
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != config:
            raise SystemExit(f"{path}의 설정과 다르다: {old} != {config}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="짝 대국 실행 (평가 기준 문서 5.3절)")
    parser.add_argument("--model", required=True, help="평가 대상 모델 ID (base, R1, ...)")
    parser.add_argument("--opponent", required=True, help="상대: base 또는 sf")
    parser.add_argument("--openings", default="0-199", help="오프닝 번호 범위 (예: 0-199, 0-9)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=None, help="Stockfish 노드 수 (상대가 sf일 때 필수)")
    parser.add_argument("--base", type=Path, default=Path("base_model/136M.pt"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/gpu"))
    parser.add_argument("--openings-file", type=Path, default=OPENINGS_PATH)
    parser.add_argument("--concurrent", type=int, default=64, help="동시 대국 수")
    parser.add_argument("--sf-workers", type=int, default=4, help="Stockfish 프로세스 수")
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.opponent not in (BASE_ID, SF_ID):
        raise SystemExit("상대는 base 또는 sf다")
    if args.model == args.opponent:
        raise SystemExit("평가 대상과 상대가 같다")
    if args.opponent == SF_ID and not args.nodes:
        raise SystemExit("상대가 sf이면 --nodes가 필요하다")

    adapters = {mid: args.runs_dir / mid / "adapter.pt" for mid in (args.model, args.opponent) if mid not in (BASE_ID, SF_ID)}
    for mid, path in adapters.items():
        if not path.exists():
            raise SystemExit(f"{mid}의 어댑터가 없다: {path}")
    indices = parse_range(args.openings)
    config = {"model": args.model, "opponent": args.opponent, "nodes": args.nodes if args.opponent == SF_ID else None,
              "openings_file": str(args.openings_file), "base": str(args.base)}
    check_config(args.out, config)

    bank = PolicyBank(args.base, adapters, device=args.device)
    pool = StockfishPool(args.nodes, args.sf_workers, args.stockfish) if args.opponent == SF_ID else None
    try:
        specs = pair_specs(load_openings(args.openings_file), indices, args.model, args.opponent)
        device_name = torch.cuda.get_device_name(0) if bank.device.type == "cuda" else platform.processor() or "cpu"
        meta = {"model": args.model, "opponent": args.opponent, "nodes": config["nodes"], "openings": args.openings,
                "device": device_name, "host": platform.node(), "torch": torch.__version__, "git_head": git_head(),
                "sha256": bank.sha256, "sf_workers": args.sf_workers if pool else 0,
                **({"engine": pool.headers} if pool else {})}
        print(json.dumps(meta, ensure_ascii=False), flush=True)
        summary = run(specs, bank, args.out, opponent=pool, concurrent=args.concurrent, meta=meta)
    finally:
        if pool is not None:
            pool.close()
    print(json.dumps({k: summary[k] for k in ("requested", "skipped", "played", "seconds", "model_moves",
                                              "model_seconds", "sf_moves", "sf_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
