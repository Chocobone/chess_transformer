"""대국 앱의 추론 엔진. 모델 id와 FEN을 받아 학습과 같은 정책 π(a|s) = softmax(q / T)를 계산하고 수를 고른다.

π는 학습·평가 코드와 같은 함수(`legal_action_tokens`, `policy_from_q`)와 같은 온도로 계산하므로,
앱에서 보는 확률이 보고서의 평가값과 같은 정의다. 수는 π에서 샘플링한다 (학습 정책 그대로).

GPU에는 베이스 한 크기만 올린다. 136M에는 LoRA를 한 번만 주입하고 어댑터마다 LoRA 가중치만 바꿔 끼운다.
"베이스 136M"은 LoRA 가중치를 모두 0으로 끼운 상태다 (보정이 0이므로 출력이 베이스와 같다).
웹과 무관한 모듈이며 스레드 안전하지 않다. 동시 호출은 부르는 쪽이 막는다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess
import torch

from src.model.actions import MOVE_TO_ACTION, ordered_legal_moves
from src.model.lora import inject_lora, load_lora, lora_state_dict
from src.model.transformer import ActionValueTransformer, load_base
from src.players import PLAYERS
from src.train.policy import legal_action_tokens, policy_from_q

TEMPERATURE = 0.02  # 학습 설정 (runs/*_136M/config.json)과 같음
BASE_SIZES = ("9M", "136M", "270M")
ADAPTER_SIZE = "136M"
PLAYER_LABELS = {"tal": "탈", "karpov": "카르포프", "smyslov": "스미슬로프"}
COMPARE_IDS = ("base-136M", "tal-136M", "karpov-136M", "smyslov-136M")


@dataclass(frozen=True)
class ModelSpec:
    id: str  # "base-136M", "tal-136M" 등
    label: str  # 화면 표시용
    size: str  # BASE_SIZES 중 하나
    base: Path  # 베이스 체크포인트
    adapter: Path | None = None  # 베이스만 쓰면 None


@dataclass(frozen=True)
class Candidate:
    uci: str
    san: str
    prob: float  # π(a|s)
    q: float  # q(s, a): 둘 차례인 쪽의 예상 승률


@dataclass(frozen=True)
class Analysis:
    model_id: str
    candidates: list[Candidate]  # π 내림차순 상위 k개
    win_prob: float  # max_a q(s, a)
    legal_count: int


def default_specs(base_dir: str | Path = "base_model", runs_dir: str | Path = "runs") -> list[ModelSpec]:
    """베이스 세 크기와 136M 기사별 어댑터 가운데 파일이 있는 것만 돌려준다."""
    base_dir, runs_dir = Path(base_dir), Path(runs_dir)
    specs = []
    for size in BASE_SIZES:
        path = base_dir / f"{size}.pt"
        if path.exists():
            specs.append(ModelSpec(f"base-{size}", f"베이스 ({size})", size, path))
    base = base_dir / f"{ADAPTER_SIZE}.pt"
    for player in PLAYERS:
        adapter = runs_dir / f"{player}_{ADAPTER_SIZE}" / "adapter.pt"
        if base.exists() and adapter.exists():
            label = f"{PLAYER_LABELS.get(player, player)} ({ADAPTER_SIZE})"
            specs.append(ModelSpec(f"{player}-{ADAPTER_SIZE}", label, ADAPTER_SIZE, base, adapter))
    return specs


def parse_board(fen: str) -> chess.Board:
    """FEN → 수를 둘 수 있는 국면. 잘못된 FEN이거나 합법수가 없으면 ValueError."""
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"잘못된 FEN이다: {fen}") from error
    if not board.is_valid():
        raise ValueError(f"규칙에 맞지 않는 국면이다: {fen}")
    if board.is_game_over():
        raise ValueError(f"합법수가 없는 국면이다: {fen}")
    return board


class Engine:
    def __init__(
        self,
        specs: list[ModelSpec],
        device: str | torch.device | None = None,
        loader: Callable[[Path, torch.device], ActionValueTransformer] = load_base,
    ):
        if len({spec.id for spec in specs}) != len(specs):
            raise ValueError("모델 id가 겹친다")
        self.specs = {spec.id: spec for spec in specs}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._loader = loader

        # 어댑터 가중치는 작으므로 (136M에서 약 4 MB) 시작할 때 모두 CPU에 읽어 둔다
        self._adapters: dict[str, dict[str, torch.Tensor]] = {}
        self._settings: dict | None = None
        for spec in specs:
            if spec.adapter is None:
                continue
            if spec.size != ADAPTER_SIZE:
                raise ValueError(f"어댑터는 {ADAPTER_SIZE} 베이스에만 쓴다: {spec.id}")
            checkpoint = torch.load(spec.adapter, map_location="cpu", weights_only=True)
            if self._settings is None:
                self._settings = checkpoint["settings"]
            elif checkpoint["settings"] != self._settings:
                raise ValueError(f"어댑터 설정이 서로 다르다: {spec.id} {checkpoint['settings']} != {self._settings}")
            self._adapters[spec.id] = checkpoint["state_dict"]
        bases = {spec.base for spec in specs if spec.size == ADAPTER_SIZE}
        if self._adapters and len(bases) != 1:
            raise ValueError(f"{ADAPTER_SIZE} 베이스 경로가 하나가 아니다: {sorted(map(str, bases))}")

        self._model: ActionValueTransformer | None = None
        self._size: str | None = None
        self._zero: dict[str, torch.Tensor] | None = None  # "베이스 136M"용 LoRA 가중치 (모두 0)
        self._active: str | None = None  # 지금 끼워진 LoRA 가중치가 어느 모델 id의 것인지

    def available(self) -> list[ModelSpec]:
        return list(self.specs.values())

    def analyze(self, model_id: str, fen: str, top_k: int = 5) -> Analysis:
        """국면의 합법수 전체를 평가해 π 상위 k개 후보를 돌려준다."""
        return self._evaluate(model_id, fen, top_k)[0]

    def choose(self, model_id: str, fen: str, generator: torch.Generator, top_k: int = 5) -> tuple[Candidate, Analysis]:
        """π에서 수 하나를 샘플링한다. 고른 수와 상위 k개 후보를 함께 돌려준다."""
        analysis, moves, sans, pi, q = self._evaluate(model_id, fen, top_k)
        index = int(torch.multinomial(pi, 1, generator=generator))
        return Candidate(moves[index].uci(), sans[index], float(pi[index]), float(q[index])), analysis

    def compare(self, model_ids: list[str], fen: str, top_k: int = 3) -> list[Analysis]:
        """같은 국면을 여러 모델로 평가한다. 결과는 요청한 순서다."""
        for model_id in model_ids:
            self._spec(model_id)
        return [self.analyze(model_id, fen, top_k) for model_id in model_ids]

    def _spec(self, model_id: str) -> ModelSpec:
        if model_id not in self.specs:
            raise ValueError(f"없는 모델이다: {model_id}")
        return self.specs[model_id]

    def _evaluate(self, model_id: str, fen: str, top_k: int):
        spec = self._spec(model_id)
        board = parse_board(fen)
        moves = ordered_legal_moves(board)
        model = self._prepare(spec)

        tokens, owner, action_ids = legal_action_tokens([board.fen()])
        assert action_ids.tolist() == [MOVE_TO_ACTION[move.uci()] for move in moves]
        with torch.inference_mode():
            q = model.win_prob(tokens.to(self.device)).float()
            log_pi, _ = policy_from_q(q, owner.to(self.device), TEMPERATURE)
        q, pi = q.cpu(), log_pi[0].exp().cpu()

        sans = [board.san(move) for move in moves]
        order = torch.argsort(pi, descending=True, stable=True)[:top_k].tolist()
        candidates = [Candidate(moves[i].uci(), sans[i], float(pi[i]), float(q[i])) for i in order]
        analysis = Analysis(model_id, candidates, float(q.max()), len(moves))
        return analysis, moves, sans, pi, q

    def _prepare(self, spec: ModelSpec) -> ActionValueTransformer:
        """spec의 베이스를 GPU에 올리고 (다른 크기는 내린다), 136M이면 맞는 LoRA 가중치를 끼운다."""
        if self._size != spec.size:
            self._unload()
            model = self._loader(spec.base, self.device).eval()
            if spec.size == ADAPTER_SIZE and self._settings is not None:
                s = self._settings
                inject_lora(model, s["rank"], s["alpha"], s["dropout"], tuple(s["targets"]))
                self._zero = {key: torch.zeros_like(value) for key, value in lora_state_dict(model).items()}
            self._model, self._size = model, spec.size

        if self._zero is not None and self._active != spec.id:
            load_lora(self._model, self._adapters.get(spec.id, self._zero))
            self._active = spec.id
        return self._model

    def _unload(self) -> None:
        self._model = self._size = self._zero = self._active = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
