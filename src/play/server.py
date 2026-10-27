"""대국 앱 서버. 로컬 한 사용자용이므로 진행 중인 대국은 메모리에 하나만 둔다.

실행: python -m src.play.server [--host 127.0.0.1] [--port 8000] [--device cuda] [--seed N]
"""

import argparse
import secrets
import threading
from dataclasses import asdict
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.play.engine import COMPARE_IDS, Engine, default_specs
from src.play.game import Game, TurnError

STATIC_DIR = Path(__file__).parent / "static"
CANDIDATES = 5  # 모델 수와 함께 보내는 후보 수
COMPARE_CANDIDATES = 3  # 비교 패널의 모델별 후보 수


class NewGame(BaseModel):
    model: str
    color: str = "white"  # "white" | "black" | "random"


class UserMove(BaseModel):
    uci: str


class AnalyzeRequest(BaseModel):
    fen: str
    models: list[str] | None = None  # 없으면 비교 패널 기본 모델


def create_app(engine: Engine, seed: int) -> FastAPI:
    app = FastAPI(title="체스 대국 앱")
    lock = threading.Lock()  # 엔진과 대국 상태는 한 번에 한 요청만 다룬다
    generator = torch.Generator().manual_seed(seed)
    labels = {spec.id: spec.label for spec in engine.available()}
    compare_ids = [model_id for model_id in COMPARE_IDS if model_id in labels]
    current: dict[str, Game | None] = {"game": None}

    def game() -> Game:
        if current["game"] is None:
            raise HTTPException(404, "진행 중인 대국이 없다")
        return current["game"]

    def guarded(action):
        """엔진·대국의 오류를 HTTP 오류로 바꾼다. 규칙 위반 400, 차례·종료 409."""
        try:
            return action()
        except TurnError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, f"화면 파일이 없다: {page}")
        return FileResponse(page)

    @app.get("/api/models")
    def models():
        return {
            "models": [{"id": s.id, "label": s.label, "size": s.size} for s in engine.available()],
            "compare": compare_ids,
        }

    @app.post("/api/game")
    def new_game(request: NewGame):
        if request.model not in labels:
            raise HTTPException(400, f"없는 모델이다: {request.model}")
        if request.color not in ("white", "black", "random"):
            raise HTTPException(400, f"잘못된 색이다: {request.color}")
        with lock:
            if request.color == "random":
                color = bool(torch.randint(2, (1,), generator=generator))
            else:
                color = request.color == "white"
            current["game"] = Game(request.model, labels[request.model], color)
            return current["game"].state()

    @app.get("/api/game")
    def get_game():
        with lock:
            return game().state()

    @app.post("/api/game/move")
    def user_move(request: UserMove):
        with lock:
            g = game()
            guarded(lambda: g.push_user(request.uci))
            return g.state()

    @app.post("/api/game/engine")
    def engine_move():
        with lock:
            g = game()

            def play():
                g.require_turn(user=False)
                move, analysis = engine.choose(g.model_id, g.board.fen(), generator, CANDIDATES)
                g.push_model(move, analysis)

            guarded(play)
            return g.state()

    @app.post("/api/game/undo")
    def undo():
        with lock:
            g = game()
            g.undo()
            return g.state()

    @app.post("/api/game/resign")
    def resign():
        with lock:
            g = game()
            guarded(g.resign)
            return g.state()

    @app.get("/api/game/pgn")
    def pgn():
        with lock:
            g = game()
            name = f"game_{g.started:%Y%m%d}_{g.model_id}.pgn"
            return PlainTextResponse(g.pgn(), headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest):
        ids = request.models if request.models is not None else compare_ids
        with lock:
            results = guarded(lambda: engine.compare(ids, request.fen, COMPARE_CANDIDATES))
        return {"fen": request.fen, "results": [{**asdict(r), "label": labels[r.model_id]} for r in results]}

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="모델과 사용자가 대국하는 로컬 웹 앱")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="기본: CUDA가 있으면 cuda")
    parser.add_argument("--seed", type=int, default=None, help="모델 수 샘플링 시드. 기본: 매번 무작위")
    parser.add_argument("--base-dir", default="base_model")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    specs = default_specs(args.base_dir, args.runs_dir)
    if not specs:
        raise SystemExit(f"모델 파일이 없다: {args.base_dir}/*.pt, {args.runs_dir}/*_136M/adapter.pt")
    engine = Engine(specs, args.device)
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    print(f"모델 {len(specs)}개: {', '.join(spec.id for spec in specs)}")
    print(f"장치 {engine.device}, 샘플링 시드 {seed}")
    print(f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")

    import uvicorn

    uvicorn.run(create_app(engine, seed), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
