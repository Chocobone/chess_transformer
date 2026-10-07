import chess
import pytest
import torch

from src.model.lora import LoRALinear, inject_lora, load_lora, lora_state_dict
from src.model.transformer import ActionValueTransformer, TransformerConfig, encode_position

TINY = TransformerConfig(embedding_dim=32, num_layers=2, num_heads=4)


def tiny_model(seed: int = 0) -> ActionValueTransformer:
    torch.manual_seed(seed)
    return ActionValueTransformer(TINY).eval()


def start_tokens() -> torch.Tensor:
    return torch.from_numpy(encode_position(chess.STARTING_FEN)[0])


def randomize_lora(model: ActionValueTransformer) -> None:
    """B가 0이면 어댑터가 출력에 영향을 주지 않으므로, 영향을 보려는 테스트에서는 B를 채운다."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.1)


def test_injection_keeps_output():
    model = tiny_model()
    tokens = start_tokens()
    before = model(tokens)
    inject_lora(model, rank=4, alpha=8)
    assert torch.equal(model(tokens), before)


def test_only_lora_is_trainable():
    model = tiny_model()
    parameters = inject_lora(model, rank=4, alpha=8)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    # 블록 2개 × q/k/v/o × (A, B)
    assert len(parameters) == len(trainable) == 2 * 4 * 2
    assert sum(parameter.numel() for parameter in parameters) == 2 * 4 * 2 * 4 * 32


def test_target_selection():
    model = tiny_model()
    inject_lora(model, rank=2, targets=("q", "v"))
    block = model.blocks[0]
    assert isinstance(block.attn.q, LoRALinear) and isinstance(block.attn.v, LoRALinear)
    assert not isinstance(block.attn.k, LoRALinear) and not isinstance(block.mlp.out, LoRALinear)

    model = tiny_model()
    parameters = inject_lora(model, rank=2, targets=("q", "k", "v", "o", "mlp"))
    assert len(parameters) == 2 * 7 * 2
    assert isinstance(model.blocks[1].mlp.split_1, LoRALinear)

    with pytest.raises(ValueError):
        inject_lora(tiny_model(), targets=("q", "head"))
    with pytest.raises(ValueError):
        inject_lora(model)  # 두 번 주입


def test_training_step_leaves_base_untouched():
    model = tiny_model()
    base = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = inject_lora(model, rank=4, alpha=8)
    randomize_lora(model)
    adapter_before = {key: value.clone() for key, value in lora_state_dict(model).items()}

    optimizer = torch.optim.SGD(parameters, lr=0.1)
    model.win_prob(start_tokens()).sum().backward()
    optimizer.step()

    after = model.state_dict()
    for name, value in base.items():
        # q/k/v/o의 원래 가중치는 `.base.` 아래로 옮겨진다
        moved = name.replace(".weight", ".base.weight") if name not in after else name
        assert torch.equal(after[moved], value), name
    adapter_after = lora_state_dict(model)
    assert any(not torch.equal(adapter_after[key], adapter_before[key]) for key in adapter_before)


def test_save_and_load_adapter(tmp_path):
    tokens = start_tokens()
    trained = tiny_model()
    base_output = trained(tokens)
    inject_lora(trained, rank=4, alpha=8)
    randomize_lora(trained)
    expected = trained(tokens)
    assert not torch.allclose(expected, base_output)

    state = lora_state_dict(trained)
    assert all(key.endswith(("lora_A", "lora_B")) for key in state)
    torch.save(state, tmp_path / "adapter.pt")

    fresh = tiny_model()  # 같은 seed → 같은 베이스
    inject_lora(fresh, rank=4, alpha=8)
    load_lora(fresh, torch.load(tmp_path / "adapter.pt", weights_only=True))
    assert torch.equal(fresh(tokens), expected)


def test_load_rejects_mismatched_keys():
    source = tiny_model()
    inject_lora(source, rank=4, targets=("q", "v"))
    target = tiny_model()
    inject_lora(target, rank=4)
    with pytest.raises(ValueError):
        load_lora(target, lora_state_dict(source))  # 누락
    with pytest.raises(ValueError):
        load_lora(source, lora_state_dict(target))  # 잉여


def test_dropout_only_in_training_mode():
    model = tiny_model()
    inject_lora(model, rank=4, alpha=8, dropout=0.5)
    randomize_lora(model)
    tokens = start_tokens()
    model.eval()
    assert torch.equal(model(tokens), model(tokens))
    model.train()
    assert not torch.equal(model(tokens), model(tokens))
