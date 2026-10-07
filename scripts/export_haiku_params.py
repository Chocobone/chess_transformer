"""원본 searchless_chess 체크포인트(orbax) → 평평한 npz. JAX 환경(GPU 서버)에서 실행한다.

    python scripts/export_haiku_params.py --checkpoint base_model/270M --out base_model/270M_params.npz

필요 패키지: jax, orbax-checkpoint, numpy. 이 저장소의 다른 코드에는 의존하지 않는다.
출력 npz의 키는 "모듈경로/파라미터명"이며 `src/model/convert.py`의 입력이 된다.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp


def load_params(directory: Path) -> dict:
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    tree = checkpointer.metadata(directory).item_metadata.tree
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
    return checkpointer.restore(directory, args=ocp.args.PyTreeRestore(restore_args=restore_args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="체크포인트 폴더 (예: base_model/270M)")
    parser.add_argument("--step", type=int, default=None, help="기본값: 가장 큰 step")
    parser.add_argument("--ema", action="store_true",
                        help="params 대신 params_ema를 쓴다 (원본 엔진은 params를 쓴다)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    step = args.step
    if step is None:
        step = max(int(p.name) for p in args.checkpoint.iterdir() if p.name.isdigit())
    directory = (args.checkpoint / str(step) / ("params_ema" if args.ema else "params")).resolve()

    params = load_params(directory)
    flat = {f"{module}/{name}": np.asarray(value)
            for module, values in params.items() for name, value in values.items()}
    np.savez(args.out, **flat)
    print(f"{directory} → {args.out}: 텐서 {len(flat)}개, 파라미터 {sum(v.size for v in flat.values()):,}개")


if __name__ == "__main__":
    main()
