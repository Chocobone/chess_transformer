import json
import math
import random

import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.cache_base_q import build_cache, compute_base_q
from src.data.dataset import Batch, PositionDataset, cache_path, collate_fn, split_by_moves
from src.model.lora import LoRALinear, inject_lora, load_adapter, lora_state_dict
from src.model.transformer import ActionValueTransformer, TransformerConfig, load_base, save_base
from src.train.policy import legal_action_tokens
from src.train.train import Objective, accumulate_gradients, build_parser, evaluate, lr_factor, train

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)


def random_records(count: int, seed: int) -> list[dict]:
    """무작위 대국에서 뽑은 (국면, 둔 수). 전처리 출력 중 학습이 쓰는 필드만 채운다."""
    rng = random.Random(seed)
    records, board = [], chess.Board()
    while len(records) < count:
        if board.is_game_over() or board.fullmove_number > 40:
            board = chess.Board()
        move = rng.choice(list(board.legal_moves))
        records.append({"fen": board.fen(), "move": move.uci()})
        board.push(move)
    return records


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """소형 무작위 베이스, 국면 데이터, q 캐시가 들어 있는 폴더."""
    root = tmp_path_factory.mktemp("train")
    data_dir = root / "data"
    data_dir.mkdir()
    for split, count, seed in (("train", 120, 1), ("val", 40, 2)):
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for record in random_records(count, seed):
                f.write(json.dumps(record) + "\n")
    torch.manual_seed(0)
    base = root / "tiny.pt"
    save_base(ActionValueTransformer(TINY), base)
    build_cache(load_base(base).eval(), data_dir, base, batch_moves=200)
    return root, data_dir, base


def load_batch(data_dir, base, split="train", count=16) -> Batch:
    dataset = PositionDataset(data_dir, split, base)
    return collate_fn([dataset[i] for i in range(count)])


def test_cache_matches_policy_order_and_model(workspace):
    _, data_dir, base = workspace
    dataset = PositionDataset(data_dir, "val", base)
    assert len(dataset) == 40
    tokens, owner, action_ids = legal_action_tokens(dataset.fens)
    assert np.array_equal(dataset.actions, action_ids.numpy())
    assert np.array_equal(np.diff(dataset.offsets), torch.bincount(owner).numpy())
    with torch.no_grad():
        expected = load_base(base).eval().win_prob(tokens)
    assert np.allclose(dataset.q, expected.numpy(), atol=1e-5)


def test_compute_base_q_is_independent_of_batch_size(workspace):
    _, data_dir, base = workspace
    model = load_base(base).eval()
    fens = PositionDataset(data_dir, "val", base).fens
    small = compute_base_q(model, fens, batch_moves=1)  # 국면 하나씩
    large = compute_base_q(model, fens, batch_moves=10_000)
    assert np.array_equal(small[0], large[0]) and np.array_equal(small[1], large[1])
    assert np.allclose(small[2], large[2], atol=1e-5)


def test_collate(workspace):
    _, data_dir, base = workspace
    dataset = PositionDataset(data_dir, "train", base)
    batch = collate_fn([dataset[i] for i in range(8)])
    counts = [chess.Board(fen).legal_moves.count() for fen in dataset.fens[:8]]
    assert batch.num_positions == 8 and batch.num_moves == sum(counts)
    assert batch.tokens.shape == (sum(counts), 79)
    assert torch.bincount(batch.owner).tolist() == counts
    # target_idx가 가리키는 줄의 행동 토큰이 기사가 둔 수다
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    chosen = batch.tokens[torch.from_numpy(starts) + batch.target_idx, -2]
    assert chosen.tolist() == dataset.targets[:8].tolist()
    assert np.array_equal(batch.q_base.numpy(), dataset.q[: sum(counts)])


def test_dataset_rejects_stale_cache(workspace, tmp_path):
    _, data_dir, base = workspace
    for name in ("train.jsonl", "val.jsonl"):
        (tmp_path / name).write_text((data_dir / name).read_text())
    with pytest.raises(FileNotFoundError):
        PositionDataset(tmp_path, "train", base)

    with np.load(cache_path(data_dir, base)) as cache:
        arrays = dict(cache)
    # 수 순서가 뒤바뀐 캐시
    tampered = dict(arrays)
    tampered["train_actions"] = arrays["train_actions"][::-1].copy()
    np.savez(cache_path(tmp_path, base), **tampered)
    dataset = PositionDataset(tmp_path, "train", base)
    with pytest.raises(ValueError):
        collate_fn([dataset[0], dataset[1]])
    # 국면 수가 다른 캐시
    tampered = dict(arrays)
    tampered["train_offsets"] = arrays["train_offsets"][:-1]
    np.savez(cache_path(tmp_path, base), **tampered)
    with pytest.raises(ValueError):
        PositionDataset(tmp_path, "train", base)


@pytest.mark.parametrize("max_moves", [1, 30, 64, 128, 10_000])
def test_split_by_moves(workspace, max_moves):
    _, data_dir, base = workspace
    batch = load_batch(data_dir, base)
    counts = torch.bincount(batch.owner).tolist()
    chunks = split_by_moves(batch, max_moves)

    # 이어 붙이면 원래 배치가 된다: 순서와 총 개수 보존, 국면은 쪼개지지 않는다
    assert sum(chunk.num_positions for chunk in chunks) == batch.num_positions
    assert torch.equal(torch.cat([chunk.tokens for chunk in chunks]), batch.tokens)
    assert torch.equal(torch.cat([chunk.q_base for chunk in chunks]), batch.q_base)
    assert torch.equal(torch.cat([chunk.target_idx for chunk in chunks]), batch.target_idx)
    assert [n for chunk in chunks for n in torch.bincount(chunk.owner).tolist()] == counts

    for chunk in chunks:
        assert chunk.owner[0] == 0 and chunk.owner[-1] == chunk.num_positions - 1
        # 한도 이하이거나, 한도보다 큰 국면 하나만 든 묶음이다
        assert chunk.num_moves <= max_moves or chunk.num_positions == 1
    if max_moves == 1:
        assert len(chunks) == batch.num_positions
    if max_moves == 10_000:
        assert len(chunks) == 1


def lora_model(base, seed: int = 0):
    model = load_base(base)
    parameters = inject_lora(model, rank=4, alpha=8)
    # B가 0이면 A의 기울기가 0이라 비교가 약해지므로 B를 채운다
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.05)
    return model, parameters


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_accumulated_gradient_equals_single_pass(workspace, gamma):
    _, data_dir, base = workspace
    batch = load_batch(data_dir, base, count=24)
    objective = Objective(temperature=0.02, tau=0.0, lam=1.0, gamma=gamma)

    model, parameters = lora_model(base)
    single = accumulate_gradients(model, batch, objective, max_moves=10_000)
    expected = [parameter.grad.clone() for parameter in parameters]
    assert any(grad.abs().max() > 0 for grad in expected)

    for max_moves in (1, 64, 128):
        model.zero_grad()
        totals = accumulate_gradients(model, batch, objective, max_moves=max_moves)
        assert len(split_by_moves(batch, max_moves)) > 1
        for parameter, grad in zip(parameters, expected):
            scale = grad.abs().max().item() + 1e-12
            assert torch.allclose(parameter.grad, grad, atol=1e-4 * scale, rtol=1e-4)
        for name, value in single.items():
            assert totals[name] == pytest.approx(value, rel=1e-4, abs=1e-6)


def test_update_is_independent_of_chunk_size(workspace):
    _, data_dir, base = workspace
    batch = load_batch(data_dir, base, count=24)
    objective = Objective()
    adapters = []
    for max_moves in (10_000, 48):
        model, parameters = lora_model(base)
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        accumulate_gradients(model, batch, objective, max_moves=max_moves)
        optimizer.step()
        adapters.append(lora_state_dict(model))
    for key, value in adapters[0].items():
        assert torch.allclose(adapters[1][key], value, atol=1e-6), key


def test_evaluate_hand_calculation():
    # 국면 1개, 수 2개. 베이스 q = [0.6, 0.5], T = 0.1 → π = [σ(1), 1 − σ(1)]. 기사는 둘째 수를 뒀다.
    batch = Batch(
        tokens=torch.zeros(2, 79, dtype=torch.long),
        owner=torch.tensor([0, 0]),
        q_base=torch.tensor([0.6, 0.5]),
        target_idx=torch.tensor([1]),
    )
    metrics = evaluate(None, [batch], Objective(temperature=0.1, tau=0.05), max_moves=1024)
    second = 1 - 1 / (1 + math.exp(-1.0))
    assert metrics["imitation"] == pytest.approx(-math.log(second), abs=1e-5)
    assert metrics["top1"] == 0.0 and metrics["top3"] == 1.0
    assert metrics["regret"] == pytest.approx(second * 0.1, abs=1e-6)
    assert metrics["regret_top1"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["band_violation"] == pytest.approx(second, abs=1e-6)
    assert metrics["strength"] == pytest.approx(second * 0.05, abs=1e-6)


def test_untrained_adapter_matches_baseline(workspace):
    _, data_dir, base = workspace
    loader = DataLoader(PositionDataset(data_dir, "val", base), batch_size=16, collate_fn=collate_fn)
    objective = Objective()
    baseline = evaluate(None, loader, objective, max_moves=1024)
    model = load_base(base)
    inject_lora(model, rank=4)
    model.train()
    metrics = evaluate(model, loader, objective, max_moves=100)
    assert model.training  # 평가 뒤 학습 모드로 돌아온다
    for name, value in baseline.items():
        assert metrics[name] == pytest.approx(value, rel=1e-3, abs=1e-5), name


def test_lr_factor():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-12)


def test_training_smoke(workspace):
    """캐시 → 학습 몇 스텝 → 어댑터 저장 → 새 베이스에 재로드까지 CPU에서 완주한다."""
    root, data_dir, base = workspace
    out = root / "run"
    args = build_parser().parse_args([
        "--player", "tal", "--base", str(base), "--out", str(out), "--data-dir", str(data_dir),
        "--rank", "4", "--lora-alpha", "8", "--lr", "1e-2", "--epochs", "2", "--batch-positions", "16",
        "--max-moves-per-forward", "64", "--max-steps", "6", "--log-every", "2", "--eval-every", "3",
        "--device", "cpu", "--num-workers", "0",
    ])
    final = train(args)

    records = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    assert records[0]["type"] == "baseline"
    assert [r["step"] for r in records if r["type"] == "train"] == [2, 4, 6]
    assert [r["step"] for r in records if r["type"] == "eval"] == [3, 6]
    assert all(math.isfinite(r["loss"]) for r in records if r["type"] == "train")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["total_steps"] == 6 and config["trainable_parameters"] == 2 * 4 * 2 * 4 * 32

    # 어댑터만 저장된다
    checkpoint = torch.load(out / "adapter.pt", weights_only=True)
    assert all(key.endswith(("lora_A", "lora_B")) for key in checkpoint["state_dict"])

    # 새 베이스에 실으면 학습 직후와 같은 지표가 나오고, 베이스 기준선과는 다르다
    model = load_base(base)
    settings = load_adapter(model, out / "adapter.pt")
    assert settings == {"rank": 4, "alpha": 8.0, "dropout": 0.0, "targets": ["q", "k", "v", "o"]}
    loader = DataLoader(PositionDataset(data_dir, "val", base), batch_size=16, collate_fn=collate_fn)
    reloaded = evaluate(model.eval(), loader, Objective(), max_moves=1024)
    for name, value in final.items():
        assert reloaded[name] == pytest.approx(value, rel=1e-4, abs=1e-6), name
    assert reloaded["imitation"] != pytest.approx(records[0]["imitation"], abs=1e-6)


def resume_args(base, data_dir, out, *extra: str):
    """epoch당 8 갱신(국면 120개, 16개씩), 2 epoch로 16 갱신. 평가는 4, 체크포인트는 3 갱신마다."""
    return build_parser().parse_args([
        "--player", "tal", "--base", str(base), "--out", str(out), "--data-dir", str(data_dir),
        "--rank", "4", "--lora-alpha", "8", "--lr", "1e-2", "--epochs", "2", "--batch-positions", "16",
        "--max-moves-per-forward", "64", "--log-every", "1", "--eval-every", "4", "--checkpoint-every", "3",
        "--warmup-steps", "1", "--device", "cpu", "--num-workers", "0", *extra,
    ])


def run_records(out) -> list[dict]:
    """log.jsonl에서 실행마다 달라지는 값(시간)과 재개 표시를 뺀 기록."""
    records = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    return [
        {key: value for key, value in record.items() if key not in ("positions_per_sec", "elapsed")}
        for record in records if record["type"] != "resume"
    ]


def crash_at_update(monkeypatch, update: int) -> None:
    """이 실행의 update번째 갱신에서 학습이 죽게 한다."""
    calls = {"count": 0}

    def crashing(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == update:
            raise RuntimeError("중단")
        return accumulate_gradients(*args, **kwargs)

    monkeypatch.setattr("src.train.train.accumulate_gradients", crashing)


@pytest.fixture(scope="module")
def uninterrupted(workspace):
    """끊기지 않고 끝까지 돈 학습."""
    root, data_dir, base = workspace
    out = root / "resume_reference"
    train(resume_args(base, data_dir, out))
    return out


# 죽는 시점(전체 기준 갱신 번호)과 그때 남아 있는 마지막 체크포인트:
# 6 → 4 (평가 직후), 8 → 6 (epoch 중간), 9 → 8 (epoch 경계), 12 → 9 (둘째 epoch 중간), 16 → 15 (마지막 갱신 직전)
@pytest.mark.parametrize("crashes", [[6], [8], [9], [12], [16], [2], [6, 12], [5, 8, 9, 14]])
def test_resume_matches_uninterrupted(workspace, uninterrupted, tmp_path, monkeypatch, crashes):
    root, data_dir, base = workspace
    out = tmp_path / "run"
    done = 0
    for crash in crashes:
        crash_at_update(monkeypatch, crash - done)
        with pytest.raises(RuntimeError, match="중단"):
            train(resume_args(base, data_dir, out, "--resume"))
        done = torch.load(out / "checkpoint.pt", weights_only=True)["step"] if (out / "checkpoint.pt").exists() else 0
    monkeypatch.undo()
    final = train(resume_args(base, data_dir, out, "--resume"))

    expected = run_records(uninterrupted)
    assert run_records(out) == expected
    assert final == {key: value for key, value in expected[-1].items() if key not in ("type", "step")}
    reference = torch.load(uninterrupted / "adapter.pt", weights_only=True)["state_dict"]
    resumed = torch.load(out / "adapter.pt", weights_only=True)["state_dict"]
    assert reference.keys() == resumed.keys()
    assert all(torch.equal(reference[key], resumed[key]) for key in reference)
    # config.json은 재개할 때 다시 쓰지 않는다
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["total_steps"] == 16


def test_resume_of_finished_run_does_nothing(workspace, uninterrupted, tmp_path):
    root, data_dir, base = workspace
    out = tmp_path / "run"
    first = train(resume_args(base, data_dir, out, "--resume"))  # 체크포인트가 없으면 처음부터 돈다
    assert run_records(out) == run_records(uninterrupted)
    adapter = (out / "adapter.pt").read_bytes()

    again = train(resume_args(base, data_dir, out, "--resume"))
    assert again == first
    assert (out / "adapter.pt").read_bytes() == adapter
    assert run_records(out) == run_records(uninterrupted)


def test_resume_rejects_other_settings_and_overwrite(workspace, tmp_path, monkeypatch):
    root, data_dir, base = workspace
    out = tmp_path / "run"
    crash_at_update(monkeypatch, 5)
    with pytest.raises(RuntimeError, match="중단"):
        train(resume_args(base, data_dir, out))
    monkeypatch.undo()
    checkpoint = (out / "checkpoint.pt").read_bytes()

    with pytest.raises(ValueError, match="lr"):
        train(resume_args(base, data_dir, out, "--resume", "--lr", "1e-3"))
    with pytest.raises(FileExistsError):
        train(resume_args(base, data_dir, out))
    assert (out / "checkpoint.pt").read_bytes() == checkpoint

    # 결과에 영향이 없는 인자는 달라도 된다
    train(resume_args(base, data_dir, out, "--resume", "--log-every", "2"))
    assert torch.load(out / "checkpoint.pt", weights_only=True)["step"] == 16
