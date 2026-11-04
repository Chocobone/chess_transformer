#!/bin/bash
# gpu_train/inputs.hf에 고정한 리비전에서 입력 파일을 받아 저장소 루트에 두고, check_inputs.sh로 확인한다.
# 서버 로그인 노드에서 setup_env.sh 뒤에 실행한다. 먼저 그 저장소를 읽을 수 있는 토큰으로 로그인한다: hf auth login
# 사용: bash gpu_train/fetch_inputs.sh
# 이미 받은 파일 중 같은 것은 다시 받지 않는다. 입력이 갱신되면 git pull 뒤 다시 실행한다.
set -euo pipefail
cd "$(dirname "$0")/.."

source gpu_train/inputs.hf
[ -n "${HF_REPO:-}" ] && [ -n "${HF_REVISION:-}" ] || {
  echo "gpu_train/inputs.hf의 HF_REPO나 HF_REVISION이 비어 있다. 로컬에서 upload_inputs.sh를 실행하고 커밋해야 한다" >&2
  exit 1
}
if [ -f gpu_train/site.env ]; then
  source gpu_train/site.env
  if [ -n "${CONDA_SH:-}" ]; then
    set +u   # conda 스크립트는 정의되지 않은 변수를 참조한다
    source "$CONDA_SH" && conda activate "$CONDA_ENV"
    set -u
  fi
fi

echo "== $HF_REPO 리비전 $HF_REVISION (${HF_TAG:-태그 없음})에서 받기"
python - "$HF_REPO" "$HF_REVISION" <<'EOF'
import filecmp
import sys
import tempfile
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download, snapshot_download

repo, revision = sys.argv[1:]
if get_token() is None:
    sys.exit("Hugging Face 토큰이 없다. 저장소를 읽을 수 있는 토큰으로 hf auth login을 먼저 하거나 HF_TOKEN을 설정한다")

# 리비전의 매니페스트가 git의 매니페스트와 같아야 한다 (다르면 inputs.hf와 inputs.sha256이 어긋난 것)
manifest = "gpu_train/inputs.sha256"
with tempfile.TemporaryDirectory() as scratch:
    remote = hf_hub_download(repo, manifest, revision=revision, repo_type="model", local_dir=scratch)
    if not filecmp.cmp(remote, manifest, shallow=False):
        sys.exit(f"{revision}의 {manifest}가 git의 {manifest}와 다르다. 입력과 매니페스트를 같은 커밋으로 맞춘다")

files = [line.split(maxsplit=1)[1] for line in Path(manifest).read_text().splitlines()
         if line.strip() and not line.startswith("#")]
snapshot_download(repo, revision=revision, repo_type="model", allow_patterns=files, local_dir=".")
print(f"받기 끝: {len(files)}개")
EOF
bash gpu_train/check_inputs.sh
