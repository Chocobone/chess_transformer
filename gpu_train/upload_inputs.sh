#!/bin/bash
# 입력 파일(gpu_train/inputs.sha256의 목록)을 Hugging Face Hub 비공개 저장소에 올리고, 올린 리비전을 gpu_train/inputs.hf에 쓴다.
# 로컬에서 실행한다. 먼저 쓰기 권한 토큰으로 로그인한다: hf auth login
# 사용: bash gpu_train/upload_inputs.sh 계정명/chess-transformer-inputs
# 끝나면 gpu_train/inputs.hf를 커밋한다. 서버는 그 리비전을 fetch_inputs.sh로 받는다.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -eq 1 ] || { echo "사용: bash gpu_train/upload_inputs.sh 계정명/저장소이름" >&2; exit 1; }
bash gpu_train/check_inputs.sh

python - "$1" <<'EOF'
import datetime
import re
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

repo = sys.argv[1]
if get_token() is None:
    sys.exit("Hugging Face 토큰이 없다. 쓰기 권한 토큰으로 hf auth login을 먼저 한다")
api = HfApi()

# 비공개 저장소에만 올린다. 이미 있는 저장소가 공개면 멈춘다
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
if not api.repo_info(repo, repo_type="model").private:
    sys.exit(f"{repo}가 공개 저장소다. 비공개로 바꾸거나 다른 저장소를 쓴다")

manifest = Path("gpu_train/inputs.sha256")
files = [line.split(maxsplit=1)[1] for line in manifest.read_text().splitlines()
         if line.strip() and not line.startswith("#")]
operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=path) for path in [*files, str(manifest)]]

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
today = datetime.date.today().strftime("%Y%m%d")
print(f"== {repo}에 {len(files)}개 + 매니페스트 올리기")
commit = api.create_commit(repo, operations=operations, repo_type="model",
                           commit_message=f"입력 {today} (chess_transformer {head or '?'})")
revision = commit.oid

# 같은 날 두 번째부터는 -2, -3 …
existing = {tag.name for tag in api.list_repo_refs(repo, repo_type="model").tags}
tag, n = f"inputs-{today}", 1
while tag in existing:
    n += 1
    tag = f"inputs-{today}-{n}"
api.create_tag(repo, tag=tag, revision=revision, repo_type="model")

path = Path("gpu_train/inputs.hf")
text = path.read_text()
for key, value in (("HF_REPO", repo), ("HF_REVISION", revision), ("HF_TAG", tag)):
    text, count = re.subn(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    assert count == 1, key
path.write_text(text, newline="\n")
print(f"== 완료: 리비전 {revision} (태그 {tag})")
print("gpu_train/inputs.hf를 커밋한다. 서버에서는 git pull 뒤 bash gpu_train/fetch_inputs.sh")
EOF
