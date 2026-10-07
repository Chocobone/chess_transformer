# 로컬 개발·테스트용 컨테이너. 호스트 GPU는 compose.yml의 GPU 예약으로 연결된다.
FROM python:3.13-slim-bookworm

ENV LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget unzip ca-certificates less procps ripgrep \
    && rm -rf /var/lib/apt/lists/*

# PyPI의 리눅스용 torch 기본 휠은 CUDA 런타임을 포함한다 (수 GB).
# 드라이버와 nvidia-smi는 이미지에 넣지 않는다. 컨테이너 실행 시 NVIDIA 런타임이 호스트 것을 연결한다.
RUN pip install torch chess numpy pytest

# 클로드 코드는 root 실행을 제한하므로 일반 사용자로 설치·실행한다
RUN useradd --create-home --shell /bin/bash dev
USER dev
ENV PATH=/home/dev/.local/bin:$PATH

RUN curl -fsSL https://claude.ai/install.sh | bash

# 저장소는 호스트에서 bind mount 되어 소유자가 다르게 보인다
RUN git config --global --add safe.directory /workspace \
    && git config --global core.quotepath false

WORKDIR /workspace
CMD ["sleep", "infinity"]
