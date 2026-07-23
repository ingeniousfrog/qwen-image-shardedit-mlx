from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from benchmark_qwen_block import (  # noqa: E402
    dequantize_weights_to_dense,
    requantize_weights,
)
from shardedit_mlx.dense_ab_profile import decide_dense_ab_verdict


def _fake_quantized_linear_weights() -> list[tuple[str, mx.array]]:
    dense = mx.ones((64, 64), dtype=mx.bfloat16)
    packed, scales, biases = mx.quantize(dense, group_size=64, bits=6, mode="affine")
    mx.eval(packed, scales, biases)
    return [
        ("attn.to_q.weight", packed),
        ("attn.to_q.scales", scales),
        ("attn.to_q.biases", biases),
        ("attn.to_q.bias", mx.zeros((64,), dtype=mx.bfloat16)),
        ("img_mod.1.weight", mx.ones((8,), dtype=mx.bfloat16)),
    ]


def test_dequantize_weights_to_dense_drops_quant_metadata() -> None:
    flat = _fake_quantized_linear_weights()
    dense = dequantize_weights_to_dense(flat, source_bits=6)
    names = {name for name, _ in dense}
    assert "attn.to_q.weight" in names
    assert "attn.to_q.bias" in names
    assert "attn.to_q.scales" not in names
    assert "attn.to_q.biases" not in names
    assert "img_mod.1.weight" in names
    weight = dict(dense)["attn.to_q.weight"]
    assert weight.dtype == mx.bfloat16
    assert weight.shape == (64, 64)


def test_requantize_bits_16_aliases_dense_path() -> None:
    flat = _fake_quantized_linear_weights()
    via_requantize = requantize_weights(flat, source_bits=6, target_bits=16)
    via_dense = dequantize_weights_to_dense(flat, source_bits=6)
    assert {name for name, _ in via_requantize} == {name for name, _ in via_dense}


def test_decide_dense_ab_verdict_thresholds() -> None:
    verdict, _ = decide_dense_ab_verdict(
        q6_median=1.0,
        dense_median=0.80,
        dense_vs_q6_max_abs_error=1.0,
        dense_vs_q6_all_finite=True,
        speedup_threshold=0.15,
    )
    assert verdict == "dequant_has_overhead"

    verdict, _ = decide_dense_ab_verdict(
        q6_median=1.0,
        dense_median=0.95,
        dense_vs_q6_max_abs_error=1.0,
        dense_vs_q6_all_finite=True,
        speedup_threshold=0.15,
    )
    assert verdict == "gemm_bandwidth_bound"

    verdict, _ = decide_dense_ab_verdict(
        q6_median=1.0,
        dense_median=0.50,
        dense_vs_q6_max_abs_error=100.0,
        dense_vs_q6_all_finite=True,
        speedup_threshold=0.15,
    )
    assert verdict == "invalid_dense_output"
