"""기사별 LoRA 어댑터 학습.

    python -m src.train.train --player tal --base base_model/270M.pt --out runs/tal --resume

갱신 단위는 국면 `--batch-positions`개다. GPU 메모리에 맞추기 위해 한 번에 순전파·역전파하는 양은
합법수 `--max-moves-per-forward`개 이하의 묶음으로 나누고, 기울기를 누적한 뒤 한 번 갱신한다.
세 기사에는 `--player`만 바꿔 같은 설정을 쓴다.

평가 때마다와 `--checkpoint-every` 갱신마다 `{out}/checkpoint.pt`를 남긴다. `--resume`으로 다시 실행하면
마지막 체크포인트에서 이어서 돌고, 결과는 끊기지 않은 학습과 같다.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler

from src.data.dataset import Batch, PositionDataset, collate_fn, player_dir, split_by_moves
from src.model.lora import ATTENTION_TARGETS, inject_lora, load_lora, lora_state_dict, save_adapter
from src.model.transformer import ActionValueTransformer, load_base
from src.players import PLAYERS
from src.train.losses import total_loss
from src.train.policy import pad_by_owner, policy_from_q


@dataclass
class Objective:
    """손실 설정. 학습과 평가가 같은 값을 쓴다."""

    temperature: float = 0.02
    tau: float = 0.02
    lam: float = 1.0
    alpha: float = 0.0
    gamma: float = 0.0


def chunk_loss(q: torch.Tensor, chunk: Batch, objective: Objective) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """한 묶음의 손실(국면 평균)과 항별 값."""
    log_pi, mask = policy_from_q(q, chunk.owner, objective.temperature)
    q_base, _ = pad_by_owner(chunk.q_base, chunk.owner)
    base_log_pi = None
    if objective.gamma > 0:
        base_log_pi, _ = policy_from_q(chunk.q_base, chunk.owner, objective.temperature)
    return total_loss(
        log_pi, mask, chunk.target_idx, q_base,
        tau=objective.tau, lam=objective.lam, alpha=objective.alpha,
        gamma=objective.gamma, base_log_pi=base_log_pi,
    )


def accumulate_gradients(
    model: ActionValueTransformer,
    batch: Batch,
    objective: Objective,
    max_moves: int,
    amp: bool = False,
) -> dict[str, float]:
    """배치를 묶음으로 나눠 차례로 역전파한다. `.grad`에 쌓이는 값은 배치 전체를 한 번에 역전파한 것과 같다.

    손실이 국면 평균이므로 묶음의 손실에 (묶음 국면 수 / 배치 국면 수)를 곱해 더하면 배치 평균이 된다.
    호출 전에 기울기를 비우고, 호출 뒤에 clipping과 갱신을 한 번만 한다.
    """
    device = model.head.weight.device
    totals = {"loss": 0.0, "imitation": 0.0, "strength": 0.0, "style": 0.0}
    for chunk in split_by_moves(batch, max_moves):
        chunk = chunk.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            q = model.win_prob(chunk.tokens)
        loss, parts = chunk_loss(q, chunk, objective)
        weight = chunk.num_positions / batch.num_positions
        (loss * weight).backward()
        totals["loss"] += loss.item() * weight
        for name, value in parts.items():
            totals[name] += value.item() * weight
    return totals


@torch.no_grad()
def evaluate(
    model: ActionValueTransformer | None,
    loader: DataLoader,
    objective: Objective,
    max_moves: int,
    amp: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """val 지표. model이 None이면 캐시의 베이스 q로 계산한다 (베이스 기준선).

    imitation        기사 수에 대한 교차엔트로피
    top1, top3       기사 수가 정책의 상위 1/3개 안에 든 비율
    regret           Σ π(a|s)·regret(s,a) — 정책대로 둘 때 베이스 기준으로 잃는 승률의 기댓값
    regret_top1      정책의 최선수 하나의 regret
    band_violation   regret이 τ를 넘는 수에 실린 확률의 합
    strength         기력 항의 값
    """
    was_training = model is not None and model.training
    if model is not None:
        model.eval()
        device = model.head.weight.device

    sums = dict.fromkeys(("imitation", "top1", "top3", "regret", "regret_top1", "band_violation", "strength"), 0.0)
    positions = 0
    for batch in loader:
        for chunk in split_by_moves(batch, max_moves):
            chunk = chunk.to(device)
            if model is None:
                q = chunk.q_base
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    q = model.win_prob(chunk.tokens)
            log_pi, mask = policy_from_q(q, chunk.owner, objective.temperature)
            q_base, _ = pad_by_owner(chunk.q_base, chunk.owner)
            best = q_base.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True).values
            regret = (best - q_base).masked_fill(~mask, 0.0)
            pi = log_pi.exp()
            target = chunk.target_idx[:, None]
            top3 = log_pi.topk(min(3, log_pi.shape[1]), dim=1).indices

            sums["imitation"] += -log_pi.gather(1, target).sum().item()
            sums["top1"] += (top3[:, :1] == target).any(dim=1).sum().item()
            sums["top3"] += (top3 == target).any(dim=1).sum().item()
            sums["regret"] += (pi * regret).sum().item()
            sums["regret_top1"] += regret.gather(1, top3[:, :1]).sum().item()
            sums["band_violation"] += (pi * (regret > objective.tau)).sum().item()
            sums["strength"] += (pi * (regret - objective.tau).clamp_min(0)).sum().item()
            positions += chunk.num_positions

    if was_training:
        model.train()
    return {name: value / max(positions, 1) for name, value in sums.items()}


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """선형 워밍업 뒤 코사인 감쇠. step은 지금까지 끝낸 갱신 횟수다."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResumableBatchSampler(Sampler):
    """`DataLoader(shuffle=True, generator=...)`와 같은 순서로 배치를 낸다.

    `skip`을 정해 두면 다음 한 번의 순회에서 앞의 그만큼을 건너뛴다 (재개한 epoch에서 이미 끝낸 배치).
    건너뛰는 배치도 난수는 똑같이 소비하므로 그 뒤의 순서는 끊기지 않은 학습과 같다.
    """

    def __init__(self, dataset, batch_size: int, generator: torch.Generator):
        self.batches = BatchSampler(RandomSampler(dataset, generator=generator), batch_size, drop_last=False)
        self.skip = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        skip, self.skip = self.skip, 0
        for index, batch in enumerate(self.batches):
            if index >= skip:
                yield batch


# 결과에 영향을 주지 않아 재개할 때 달라도 되는 인자
RESUME_FREE_ARGS = ("resume", "num_workers", "log_every", "eval_every", "checkpoint_every", "device")


def rng_state() -> dict:
    """python·numpy·torch 전역 난수 상태. `weights_only=True`로 읽을 수 있는 형태로 바꿔 담는다."""
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached_gaussian),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    """임시 파일에 쓴 뒤 바꿔치기한다. 저장 도중 중단되어도 직전 체크포인트가 남는다."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def truncate_log(path: Path, step: int) -> None:
    """log.jsonl에서 체크포인트 뒤에 쓰인 줄(과 중단으로 잘린 줄)을 버린다."""
    kept = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("step", 0) <= step:
                kept.append(json.dumps(record, ensure_ascii=False) + "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(kept)


def train(args: argparse.Namespace) -> dict[str, float]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    objective = Objective(args.temperature, args.tau, args.lam, args.alpha, args.gamma)
    if args.alpha != 0:
        print("경고: 스타일 특성이 아직 없어 --alpha는 손실에 영향을 주지 않는다")

    data_dir = args.data_dir or player_dir(args.player)
    train_set = PositionDataset(data_dir, "train", args.base)
    val_set = PositionDataset(data_dir, "val", args.base)
    if args.eval_positions:
        val_set = torch.utils.data.Subset(val_set, range(min(args.eval_positions, len(val_set))))
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = ResumableBatchSampler(train_set, args.batch_positions, generator)
    train_loader = DataLoader(
        train_set, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=args.num_workers, generator=generator, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_positions, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = load_base(args.base, device)
    settings = {"rank": args.rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                "targets": list(args.lora_targets)}
    parameters = inject_lora(model, args.rank, args.lora_alpha, args.lora_dropout, tuple(args.lora_targets))
    model.train()

    total_steps = len(train_loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        base_config=asdict(model.config), amp=amp, total_steps=total_steps, warmup_steps=warmup_steps,
        train_positions=len(train_set), val_positions=len(val_set),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )

    checkpoint_path = out / "checkpoint.pt"
    resumed = None
    if checkpoint_path.exists():
        if not args.resume:
            raise FileExistsError(f"{checkpoint_path}가 이미 있다. 이어서 돌리려면 --resume을 붙이고, 새로 시작하려면 폴더를 비운다")
        resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        different = sorted(
            key for key in config.keys() | resumed["config"].keys()
            if key not in RESUME_FREE_ARGS and config.get(key) != resumed["config"].get(key)
        )
        if different:
            raise ValueError(f"체크포인트의 설정이 지금 인자와 다르다: {different}")
        load_lora(model, resumed["lora"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        set_rng_state(resumed["rng"])
        truncate_log(out / "log.jsonl", resumed["step"])
    else:
        with open(out / "config.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    log_file = open(out / "log.jsonl", "a" if resumed else "w", encoding="utf-8", newline="\n")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    # 베이스 기준선: 학습 전에 같은 val 지표를 캐시의 베이스 q로 한 번 잰다
    eval_moves = args.eval_max_moves
    if resumed:
        log({"type": "resume", "step": resumed["step"]})
    else:
        log({"type": "baseline", **evaluate(None, val_loader, objective, eval_moves, device=device)})

    def run_eval(step: int) -> dict[str, float]:
        # 기본은 float32 평가다. bfloat16은 q를 조금 바꿔 베이스 기준선(float32 캐시)과의 비교를 흐린다
        metrics = evaluate(model, val_loader, objective, eval_moves, amp and args.eval_amp)
        log({"type": "eval", "step": step, **metrics})
        save_adapter(model, out / "adapter.pt", settings)
        return metrics

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = resumed["step"] if resumed else 0
    first_epoch = resumed["epoch"] if resumed else 0
    metrics = resumed["metrics"] if resumed else {}
    start = time.time() - (resumed["elapsed"] if resumed else 0.0)
    window = {"positions": 0, "time": time.time()}
    position = {}  # 지금 epoch, 그 epoch에서 끝낸 배치 수, 그 epoch를 시작할 때의 데이터 순서 난수 상태

    def checkpoint() -> None:
        save_checkpoint(checkpoint_path, {
            "step": step, **position, "metrics": metrics, "elapsed": time.time() - start, "config": config,
            "lora": lora_state_dict(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
        })

    if step >= total_steps:
        log_file.close()
        return metrics

    for epoch in range(first_epoch, args.epochs):
        position = {"epoch": epoch, "batches_done": 0, "generator": generator.get_state()}
        if resumed and epoch == first_epoch:
            # 그 epoch를 시작하던 때의 난수로 같은 순서를 다시 만들고, 끝낸 배치는 건너뛴다
            generator.set_state(resumed["generator"])
            train_sampler.skip = resumed["batches_done"]
            position.update(batches_done=resumed["batches_done"], generator=resumed["generator"])
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            totals = accumulate_gradients(model, batch, objective, args.max_moves_per_forward, amp)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip).item()
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            step += 1
            position["batches_done"] += 1
            window["positions"] += batch.num_positions

            if step % args.log_every == 0 or step == total_steps:
                now = time.time()
                record = {"type": "train", "step": step, "epoch": epoch, "lr": lr, "grad_norm": grad_norm,
                          **totals, "positions_per_sec": window["positions"] / (now - window["time"]),
                          "elapsed": now - start}
                if device.type == "cuda":
                    record["max_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
                log(record)
                window = {"positions": 0, "time": now}
            if step % args.eval_every == 0 and step < total_steps:
                metrics = run_eval(step)
                checkpoint()
                window["time"] = time.time()
            elif args.checkpoint_every and step % args.checkpoint_every == 0 and step < total_steps:
                checkpoint()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    metrics = run_eval(step)
    checkpoint()
    log_file.close()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--player", required=True, choices=sorted(PLAYERS))
    parser.add_argument("--base", required=True, help="convert.py가 만든 .pt (q 캐시도 이 이름으로 찾는다)")
    parser.add_argument("--out", required=True, help="어댑터·설정·로그를 저장할 폴더")
    parser.add_argument("--data-dir", type=Path, default=None, help="기본값: data/processed/{player}")

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", nargs="+", default=list(ATTENTION_TARGETS),
                        choices=[*ATTENTION_TARGETS, "mlp"])

    parser.add_argument("--temperature", type=float, default=0.02, help="정책 π = softmax(q / T)의 T")
    parser.add_argument("--lam", type=float, default=1.0, help="기력 항 가중치 λ")
    parser.add_argument("--tau", type=float, default=0.02, help="기력 항의 허용 regret 밴드 τ")
    parser.add_argument("--alpha", type=float, default=0.0, help="스타일 항 가중치 (이번 타스크에서는 비활성)")
    parser.add_argument("--gamma", type=float, default=0.0, help="모방 항 가중치 (1 − π_base(a*))^γ의 γ")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-positions", type=int, default=32, help="갱신 1회에 쓰는 국면 수")
    parser.add_argument("--max-moves-per-forward", type=int, default=128,
                        help="한 번에 순전파·역전파하는 합법수 수의 한도 (GPU 메모리에 맞춘다)")
    parser.add_argument("--eval-max-moves", type=int, default=256, help="평가 때 한 번에 순전파하는 합법수 수")
    parser.add_argument("--warmup-steps", type=int, default=200, help="전체 갱신 횟수의 10%%를 넘지 않는다")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0이 아니면 이 갱신 횟수에서 멈춘다")

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-positions", type=int, default=0, help="0이 아니면 val 앞쪽 국면만 평가한다")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="이 갱신 횟수마다 재개용 체크포인트를 남긴다 (평가 때는 항상 남긴다). 0이면 평가 때만")
    parser.add_argument("--resume", action="store_true",
                        help="--out에 checkpoint.pt가 있으면 이어서 돈다. 없으면 처음부터 시작한다")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="혼합 정밀도(bfloat16)를 끈다")
    parser.add_argument("--eval-amp", action="store_true",
                        help="평가도 bfloat16으로 한다 (빠르지만 베이스 기준선과 미세하게 어긋난다)")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
