"""대국용 묶음 정책 (Task #13 2단계).

π(a|s) = softmax(q(s, a) / 0.02)를 학습·평가·대국 앱과 같은 함수(`legal_action_tokens`, `win_prob`, `policy_from_q`)로
계산한다. 대국 앱 `src/play/engine.py`의 `Engine`과 같은 방식으로 136M 베이스를 한 번 올리고 LoRA를 한 번 주입한 뒤
모델마다 LoRA 가중치만 바꿔 끼운다. "base"는 LoRA 가중치를 모두 0으로 끼운 상태(출력이 베이스와 같다)다.

`Engine`과 다른 점은 국면 여러 개를 묶어 한 번에 계산한다는 것뿐이다 (구현 계획서 해석 10).
"""

import hashlib
from pathlib import Path
from typing import Callable

import chess
import torch

from src.model.actions import MOVE_TO_ACTION, ordered_legal_moves
from src.model.lora import inject_lora, load_lora, lora_state_dict
from src.model.transformer import ActionValueTransformer, load_base
from src.train.policy import legal_action_tokens, policy_from_q

TEMPERATURE = 0.02  # 학습 설정과 같다 (src/play/engine.py의 TEMPERATURE)
BASE_ID = "base"
MAX_ROWS = 8192  # 한 번에 모델에 넣는 (국면, 수) 줄 수의 상한. 넘으면 국면 단위로 나눠 계산한다


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PolicyBank:
    def __init__(self, base: str | Path, adapters: dict[str, str | Path], device: str | torch.device = "cpu",
                 loader: Callable[[Path, torch.device], ActionValueTransformer] = load_base, max_rows: int = MAX_ROWS):
        if BASE_ID in adapters:
            raise ValueError(f"모델 ID {BASE_ID}는 베이스 전용이다")
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.sha256 = {BASE_ID: sha256(Path(base))} if Path(base).exists() else {BASE_ID: ""}
        self._weights: dict[str, dict[str, torch.Tensor]] = {}
        settings = None
        for model_id, path in adapters.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if settings is None:
                settings = checkpoint["settings"]
            elif checkpoint["settings"] != settings:
                raise ValueError(f"어댑터 설정이 서로 다르다: {model_id} {checkpoint['settings']} != {settings}")
            self._weights[model_id] = checkpoint["state_dict"]
            self.sha256[model_id] = sha256(Path(path))

        self.model = loader(Path(base), self.device).eval()
        if settings is not None:
            inject_lora(self.model, settings["rank"], settings["alpha"], settings["dropout"], tuple(settings["targets"]))
            self._weights[BASE_ID] = {key: torch.zeros_like(value) for key, value in lora_state_dict(self.model).items()}
        self.settings = settings
        self._active: str | None = None

    @property
    def ids(self) -> list[str]:
        return [BASE_ID, *[key for key in self._weights if key != BASE_ID]]

    def _use(self, model_id: str) -> None:
        if model_id not in self.ids:
            raise ValueError(f"없는 모델이다: {model_id}")
        if self._weights and self._active != model_id:
            load_lora(self.model, self._weights[model_id])
            self._active = model_id

    def probabilities(self, model_id: str, boards: list[chess.Board]) -> list[torch.Tensor]:
        """국면마다 합법수(`ordered_legal_moves` 순서)의 π [CPU float32]."""
        self._use(model_id)
        out: list[torch.Tensor] = []
        start = 0
        while start < len(boards):
            # max_rows를 넘지 않는 만큼 국면을 묶는다 (최소 1국면)
            rows, end = 0, start
            while end < len(boards) and (end == start or rows + boards[end].legal_moves.count() <= self.max_rows):
                rows += boards[end].legal_moves.count()
                end += 1
            tokens, owner, _ = legal_action_tokens([board.fen() for board in boards[start:end]])
            with torch.inference_mode():
                q = self.model.win_prob(tokens.to(self.device)).float()
                log_pi, mask = policy_from_q(q, owner.to(self.device), TEMPERATURE)
            pi = log_pi.exp().cpu()
            counts = mask.sum(dim=1).cpu().tolist()
            out.extend(pi[i, :count] for i, count in enumerate(counts))
            start = end
        return out

    def moves(self, model_id: str, boards: list[chess.Board], generators: list[torch.Generator]
              ) -> list[tuple[chess.Move, float]]:
        """국면마다 π에서 수 하나를 샘플링한다 (대국 앱 `Engine.choose`와 같은 `torch.multinomial`). (수, π)."""
        result = []
        for board, pi, generator in zip(boards, self.probabilities(model_id, boards), generators, strict=True):
            moves = ordered_legal_moves(board)
            assert len(moves) == len(pi)
            index = int(torch.multinomial(pi, 1, generator=generator))
            result.append((moves[index], float(pi[index])))
        return result


def check_order(board: chess.Board) -> None:
    """`legal_action_tokens`의 행동 순서가 `ordered_legal_moves`와 같은지 (테스트용)."""
    _, _, action_ids = legal_action_tokens([board.fen()])
    assert action_ids.tolist() == [MOVE_TO_ACTION[move.uci()] for move in ordered_legal_moves(board)]
