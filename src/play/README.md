# 대국 웹 앱 (`src.play`) 실행 방법

학습한 모델과 브라우저에서 직접 두어 보는 로컬 웹 앱이다 (Task #7). 화면 사용법, 응답 시간, 알려진 한계 등 상세 내용은 [`mydocs/manual/play_app.md`](../../mydocs/manual/play_app.md)에 있다.

> 모든 명령은 **저장소 루트**에서 실행한다.

## 1. 환경 설치

```bash
pip install -r requirements.txt   # fastapi, uvicorn 포함 (torch는 CUDA 버전에 맞는 휠을 먼저 설치)
pytest                            # 전부 통과해야 한다
```

GPU가 없어도 실행된다 (CPU에서는 모델 수가 느리다).

## 2. 필요한 파일

서버는 시작할 때 아래 파일을 찾아, **있는 것만** 모델 목록에 올린다.

| 모델 | 파일 |
|---|---|
| 베이스 9M · 136M · 270M | `base_model/{9M,136M,270M}.pt` |
| 탈 · 카르포프 · 스미슬로프 (136M) | `base_model/136M.pt` + `runs/{tal,karpov,smyslov}_136M/adapter.pt` |

- 베이스 가중치 준비는 [`training_guide.md`](../../mydocs/manual/training_guide.md) 2절을 따른다.
- 파일이 하나도 없으면 `모델 파일이 없다: ...`를 출력하고 종료한다. 다른 위치에 있으면 `--base-dir`, `--runs-dir`로 지정한다.

## 3. 서버 실행

```bash
python -m src.play.server                    # 같은 컴퓨터에서 접속할 때
python -m src.play.server --host 0.0.0.0     # 개발 컨테이너·WSL2 안에서 띄우고 Windows 브라우저로 접속할 때
```

정상 시작 시 출력 예:

```
모델 6개: base-9M, base-136M, base-270M, tal-136M, karpov-136M, smyslov-136M
장치 cuda, 샘플링 시드 1234567
http://localhost:4321
```

브라우저에서 **`http://localhost:4321`** 을 연다. 종료는 Ctrl+C (진행 중인 대국은 메모리에만 있으므로 사라진다. 필요하면 먼저 PGN으로 저장한다).

### 옵션

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `--host` | `127.0.0.1` | 듣는 주소. 다른 곳에서 접속하려면 `0.0.0.0` |
| `--port` | `4321` | 포트 |
| `--device` | CUDA가 있으면 `cuda` | `cpu`로 강제할 수 있다 |
| `--seed` | 무작위 (시작 시 출력) | 모델 수 샘플링·무작위 색 시드. 같은 시드와 같은 수순이면 같은 대국이 나온다 |
| `--base-dir` | `base_model` | 베이스 가중치 폴더 |
| `--runs-dir` | `runs` | 어댑터 폴더 |

## 4. 환경별 접속

### 개발 컨테이너(`chess-dev`)

`docker/compose.yml`이 컨테이너의 4321 포트를 호스트의 `127.0.0.1:4321`로 공개한다. 포트 공개 설정 이전에 만든 컨테이너라면 한 번 다시 만든다.

```bash
docker compose -f docker/compose.yml up -d     # 호스트에서 (홈 볼륨은 유지)
docker exec -it chess-dev bash
python -m src.play.server --host 0.0.0.0       # 컨테이너 안에서
```

### WSL2

`--host 0.0.0.0`으로 띄우면 보통 Windows 브라우저의 `http://localhost:4321`로 접속된다. 안 되면 WSL 안에서 `hostname -I`로 IP를 확인해 `http://<그 IP>:4321`로 연다.

## 5. 자주 나는 오류

| 증상 | 원인과 해결 |
|---|---|
| `모델 파일이 없다: ...`로 종료 | 2절의 파일이 없다. 경로가 다르면 `--base-dir`, `--runs-dir` |
| 브라우저 접속 불가 (컨테이너·WSL2) | `--host 0.0.0.0`으로 띄웠는지, 컨테이너가 포트 공개 후 만들어졌는지 확인 |
| `address already in use` | 이미 서버가 떠 있다. 끄거나 `--port`를 바꾼다 |
| "서버에 연결하지 못했습니다" 알림 | 서버가 꺼졌다. 다시 띄우고 새로고침한다 |
| 크기를 바꾼 뒤 첫 수가 몇 초 걸림 | 정상. 베이스를 불러오는 시간이다 |
| CUDA out of memory | 다른 작업이 GPU를 쓰고 있다. 그 작업을 끝내거나 `--device cpu` |

## 구성 파일

| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI 서버, 실행 진입점 (`main`) |
| `game.py` | 대국 상태 관리 |
| `engine.py` | 모델 불러오기와 수 선택 (π = softmax(q / 0.02)에서 샘플링) |
| `static/` | 브라우저 화면 (HTML·JS·CSS, cburnett 기물) |
