"""Haiku 파라미터(npz) → PyTorch 베이스 체크포인트.

    python -m src.model.convert --params base_model/270M_params.npz --size 270M --out base_model/270M.pt

입력 npz는 `scripts/export_haiku_params.py`가 원본 체크포인트에서 뽑은 것으로,
키는 "모듈경로/파라미터명" (예: "multi_head_dot_product_attention_1/linear_2/w")이다.
"""

import argparse

import numpy as np
import torch

from src.model.transformer import (
    MODEL_SIZES,
    ActionValueTransformer,
    TransformerConfig,
    config_for,
    save_base,
)


def _suffix(index: int) -> str:
    """Haiku는 같은 이름의 모듈에 생성 순서대로 "", "_1", "_2", ...를 붙인다."""
    return "" if index == 0 else f"_{index}"


def key_mapping(config: TransformerConfig) -> dict[str, tuple[str, bool]]:
    """state_dict 키 → (Haiku 키, 전치 여부). Haiku 선형층 가중치는 [in, out]이라 전치한다."""
    mapping = {
        "token_embedding.weight": ("embed/embeddings", False),
        "position_embedding.weight": ("embed_1/embeddings", False),
    }
    for i in range(config.num_layers):
        block = f"blocks.{i}"
        attention = f"multi_head_dot_product_attention{_suffix(i)}"
        for j, name in enumerate(["q", "k", "v", "o"]):
            mapping[f"{block}.attn.{name}.weight"] = (f"{attention}/linear{_suffix(j)}/w", True)
        for j, name in enumerate(["split_1", "split_2", "out"]):
            mapping[f"{block}.mlp.{name}.weight"] = (f"linear{_suffix(3 * i + j)}/w", True)
        for j, name in enumerate(["ln_attn", "ln_mlp"]):
            layer_norm = f"layer_norm{_suffix(2 * i + j)}"
            mapping[f"{block}.{name}.weight"] = (f"{layer_norm}/scale", False)
            mapping[f"{block}.{name}.bias"] = (f"{layer_norm}/offset", False)

    layer_norm = f"layer_norm{_suffix(2 * config.num_layers)}"
    head = f"linear{_suffix(3 * config.num_layers)}"
    mapping["ln_final.weight"] = (f"{layer_norm}/scale", False)
    mapping["ln_final.bias"] = (f"{layer_norm}/offset", False)
    mapping["head.weight"] = (f"{head}/w", True)
    mapping["head.bias"] = (f"{head}/b", False)
    return mapping


def convert(params: dict[str, np.ndarray], config: TransformerConfig) -> ActionValueTransformer:
    """매핑되지 않은 키, 남는 키, 형상이 다른 텐서가 하나라도 있으면 실패한다."""
    model = ActionValueTransformer(config)
    expected = model.state_dict()
    mapping = key_mapping(config)

    if mapping.keys() != expected.keys():
        raise ValueError(f"매핑과 모델 키 불일치: {sorted(mapping.keys() ^ expected.keys())}")
    haiku_keys = {haiku_key for haiku_key, _ in mapping.values()}
    if haiku_keys != params.keys():
        missing = sorted(haiku_keys - params.keys())
        unused = sorted(params.keys() - haiku_keys)
        raise ValueError(f"체크포인트 키 불일치. 없는 키: {missing}, 남는 키: {unused}")

    state_dict = {}
    for key, (haiku_key, transpose) in mapping.items():
        value = torch.from_numpy(np.asarray(params[haiku_key], dtype=np.float32))
        value = value.T.contiguous() if transpose else value
        if value.shape != expected[key].shape:
            raise ValueError(
                f"형상 불일치 {key} ← {haiku_key}: {tuple(value.shape)} != {tuple(expected[key].shape)}"
            )
        state_dict[key] = value
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", required=True, help="export_haiku_params.py가 만든 npz")
    parser.add_argument("--size", required=True, choices=sorted(MODEL_SIZES))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with np.load(args.params) as params:
        model = convert(dict(params), config_for(args.size))
    save_base(model, args.out)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{args.size}: 파라미터 {num_params:,}개 → {args.out}")


if __name__ == "__main__":
    main()
