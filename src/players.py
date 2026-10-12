"""기사별 설정. 기사 사이의 차이는 여기 있는 값뿐이며 나머지 코드는 세 기사에 대칭으로 적용된다."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Player:
    pgn: str  # 저장소 루트 기준 PGN 경로
    name_prefix: str  # White/Black 태그에서 이 기사를 식별하는 접두어

    def matches(self, name: str) -> bool:
        return name.startswith(self.name_prefix)


# 접두어에 쉼표를 포함한다. "Karpov,"는 "Karpov, Anatoly" / "Karpov,Ana" / "Karpov,An"을
# 모두 잡고 "Kasparov, Gary"는 잡지 않는다.
PLAYERS = {
    "tal": Player(pgn="player_pgn/Tal.pgn", name_prefix="Tal,"),
    "karpov": Player(pgn="player_pgn/Karpov.pgn", name_prefix="Karpov,"),
    "smyslov": Player(pgn="player_pgn/Smyslov.pgn", name_prefix="Smyslov,"),
}
