# 로컬 개발·단위 테스트용 컨테이너 (CPU 전용). 실제 학습은 GPU 서버에서 수행한다.
FROM python:3.13-slim-bookworm

ENV LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget unzip ca-certificates less procps ripgrep \
    && rm -rf /var/lib/apt/lists/*

# torch는 CPU 휠을 쓴다 (기본 휠은 CUDA 포함으로 수 GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install chess numpy pytest

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
