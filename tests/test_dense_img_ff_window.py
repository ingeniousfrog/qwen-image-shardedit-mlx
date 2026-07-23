"""Tests for window-local dense image MLP residency helpers."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
import pytest

from shardedit_mlx.dense_img_ff_profile import (
    decide_dense_img_ff_window_verdict,
    decide_dense_prefetch_verdict,
)
from shardedit_mlx.dense_img_ff_window import (
    DenseFeedForward,
    materialize_dense_img_ff,
    materialize_dense_linear,
    prepare_dense_img_ff_window,
)
from shardedit_mlx.q6_metal_mlp import eager_q6_mlp
from shardedit_mlx.qwen_block_loader import LoadedBlock
from shardedit_mlx.residency_plan import ResidencyWindow


def _make_ff(in_features: int = 64, mid: int = 128, out_features: int = 64) -> nn.Module:
    class FF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_in = nn.QuantizedLinear.from_linear(
                nn.Linear(in_features, mid, bias=True),
                bits=6,
                group_size=64,
            )
            self.mlp_out = nn.QuantizedLinear.from_linear(
                nn.Linear(mid, out_features, bias=True),
                bits=6,
                group_size=64,
            )

        def __call__(self, hidden: mx.array) -> mx.array:
            return eager_q6_mlp(self, hidden)

    return FF()


def test_materialize_keeps_lora_additive() -> None:
    mx.random.seed(2)
    base = nn.QuantizedLinear.from_linear(nn.Linear(64, 128, bias=True), bits=6, group_size=64)
    from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

    lora = LoRALinear.from_linear(base, r=8, scale=0.5)
    lora.lora_A = mx.random.normal((64, 8)).astype(mx.bfloat16) * 0.01
    lora.lora_B = mx.random.normal((8, 128)).astype(mx.bfloat16) * 0.01

    class TinyFF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_in = lora
            self.mlp_out = nn.QuantizedLinear.from_linear(
                nn.Linear(128, 64, bias=True), bits=6, group_size=64
            )

        def __call__(self, hidden: mx.array) -> mx.array:
            hidden = self.mlp_in(hidden)
            hidden = nn.gelu_approx(hidden)
            return self.mlp_out(hidden)

    ff = TinyFF()
    dense_ff, _ = materialize_dense_img_ff(ff)
    x = mx.random.normal((4, 64)).astype(mx.bfloat16)
    y_ref = ff(x)
    y_dense = dense_ff(x)
    mx.eval(y_ref, y_dense)
    err = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_dense.astype(mx.float32))).item())
    assert dense_ff.mlp_in_lora is not None
    assert err < 0.05



def test_materialize_dense_img_ff_matches_eager() -> None:
    mx.random.seed(0)
    ff = _make_ff()
    x = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
    dense_ff, nbytes = materialize_dense_img_ff(ff)
    y_q6 = ff(x)
    y_dense = dense_ff(x)
    mx.eval(y_q6, y_dense)
    err = float(mx.max(mx.abs(y_q6.astype(mx.float32) - y_dense.astype(mx.float32))).item())
    assert isinstance(dense_ff, DenseFeedForward)
    assert nbytes > 0
    assert err < 0.05


def test_prepare_reuses_cache_without_rematerializing() -> None:
    cache: dict[int, DenseFeedForward] = {}
    block = SimpleNamespace(img_ff=_make_ff())
    loaded = LoadedBlock(block_index=7, module=block)
    prepare_dense_img_ff_window([loaded], cache=cache, cache_max_blocks=4)
    assert 7 in cache
    first = cache[7]
    block2 = SimpleNamespace(img_ff=_make_ff())
    loaded2 = LoadedBlock(block_index=7, module=block2)
    prepare_dense_img_ff_window([loaded2], cache=cache, cache_max_blocks=4)
    assert block2.img_ff is first
    assert len(cache) == 1


def test_prepare_dense_img_ff_window_replaces_modules() -> None:
    mx.random.seed(1)
    original = _make_ff()
    block = SimpleNamespace(img_ff=original)
    loaded = LoadedBlock(block_index=3, module=block)
    handles = prepare_dense_img_ff_window([loaded], reclaim_quantized=False)
    assert len(handles) == 1
    assert handles[0].block_index == 3
    assert isinstance(block.img_ff, DenseFeedForward)
    assert block.img_ff is not original


def test_verdict_helps() -> None:
    verdict, reason = decide_dense_img_ff_window_verdict(
        q6_median=1.0,
        dense_median=0.8,
        q6_peak_gib=4.0,
        dense_peak_gib=4.5,
        max_abs_error=1.0,
        all_finite=True,
    )
    assert verdict == "dense_img_ff_helps"
    assert "faster" in reason


def test_prefetch_verdict_plausible_when_materialize_fits() -> None:
    verdict, reason = decide_dense_prefetch_verdict(
        materialize_median=0.40,
        q6_img_ff_median=1.00,
        dense_img_ff_median=0.70,
        window_compute_median=2.00,
    )
    assert verdict == "prefetch_plausible"
    assert "unlock" in reason


def test_prefetch_verdict_rejects_when_materialize_dominates() -> None:
    verdict, _ = decide_dense_prefetch_verdict(
        materialize_median=3.00,
        q6_img_ff_median=1.00,
        dense_img_ff_median=0.70,
        window_compute_median=2.00,
    )
    assert verdict == "prefetch_not_worth_it"


def test_prefetch_verdict_sync_already_wins() -> None:
    verdict, _ = decide_dense_prefetch_verdict(
        materialize_median=0.10,
        q6_img_ff_median=1.00,
        dense_img_ff_median=0.70,
        window_compute_median=2.00,
    )
    assert verdict == "sync_already_wins"


def test_verdict_peak_over_budget() -> None:
    verdict, _ = decide_dense_img_ff_window_verdict(
        q6_median=1.0,
        dense_median=0.8,
        q6_peak_gib=4.0,
        dense_peak_gib=6.0,
        max_abs_error=1.0,
        all_finite=True,
        peak_budget_gib=5.0,
    )
    assert verdict == "peak_over_budget"


def test_shard_runtime_calls_prepare_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from shardedit_mlx import shard_runtime

    calls: list[tuple[int, ...]] = []

    def fake_prepare(blocks, **kwargs):  # noqa: ANN001
        calls.append(tuple(block.block_index for block in blocks))
        return ()

    monkeypatch.setattr(shard_runtime, "prepare_dense_img_ff_window", fake_prepare)
    monkeypatch.setattr(
        shard_runtime,
        "load_block_window",
        lambda _layout, indices: tuple(
            LoadedBlock(block_index=index, module=SimpleNamespace(img_ff=None))
            for index in indices
        ),
    )
    monkeypatch.setattr(
        shard_runtime,
        "apply_window_loras",
        lambda *_args, **_kwargs: (),
    )

    runtime = shard_runtime.ShardTransformerRuntime(
        layout=SimpleNamespace(plans=(None, None)),
        windows=(
            ResidencyWindow(
                index=0,
                block_indices=(0, 1),
                shards=("0.safetensors",),
            ),
        ),
        lora_sources=(),
        lora_targets=[],
        dense_img_ff=True,
    )

    def apply_block(**kwargs):  # noqa: ANN003
        hidden = kwargs["hidden_states"]
        encoder = kwargs["encoder_hidden_states"]
        return encoder, hidden

    hidden = mx.zeros((1, 2, 4))
    encoder = mx.zeros((1, 2, 4))
    # Minimal transformer stub for the prefix of __call__ before windows.
    # Patch __call__ body by invoking the window loop pieces via a thin wrapper.
    # Instead, call the window section by temporarily replacing methods used before loop.
    transformer = SimpleNamespace(
        img_in=lambda x: x,
        _compute_timestep=lambda *_a, **_k: mx.zeros((1,)),
        txt_norm=lambda x: x,
        txt_in=lambda x: x,
        time_text_embed=lambda *_a, **_k: mx.zeros((1, 4)),
        _compute_rotary_embeddings=lambda **_k: None,
        pos_embed=None,
        norm_out=lambda x, *_a: x,
        proj_out=lambda x: x,
    )
    config = SimpleNamespace()
    out = runtime(
        transformer,
        t=0,
        config=config,
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        encoder_hidden_states_mask=mx.ones((1, 2)),
        cond_image_grid=None,
        apply_block=apply_block,
    )
    mx.eval(out)
    assert calls == [(0, 1)]
