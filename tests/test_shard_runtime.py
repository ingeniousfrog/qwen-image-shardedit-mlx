from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

try:
    from shardedit_mlx import shard_runtime
except RuntimeError as error:
    if "No Metal device available" not in str(error):
        raise
    pytest.skip("MLX Metal device is not available", allow_module_level=True)

from shardedit_mlx.residency_plan import ResidencyWindow


class FakeSafeTensorHandle:
    def __init__(self, keys: tuple[str, ...]) -> None:
        self._keys = keys
        self.loaded_keys: list[str] = []

    def __enter__(self) -> FakeSafeTensorHandle:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def get_tensor(self, key: str) -> str:
        self.loaded_keys.append(key)
        return f"tensor:{key}"


def test_torch_lora_loader_reads_only_selected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = FakeSafeTensorHandle(("block.0.a", "block.0.b", "block.1.c"))

    def fake_safe_open(path: Path, **kwargs: Any) -> FakeSafeTensorHandle:
        assert path == Path("lora.safetensors")
        assert kwargs == {"framework": "pt", "device": "cpu"}
        return handle

    monkeypatch.setattr(shard_runtime, "safe_open", fake_safe_open)
    monkeypatch.setattr(
        shard_runtime,
        "_torch_tensor_to_mx",
        lambda tensor: f"mx:{tensor}",
    )

    weights = shard_runtime._load_lora_weights_with_torch(
        Path("lora.safetensors"),
        ("block.1.c", "block.0.a"),
    )

    assert handle.loaded_keys == ["block.1.c", "block.0.a"]
    assert weights == {
        "block.1.c": "mx:tensor:block.1.c",
        "block.0.a": "mx:tensor:block.0.a",
    }


def test_torch_lora_loader_rejects_missing_selected_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shard_runtime,
        "safe_open",
        lambda *_args, **_kwargs: FakeSafeTensorHandle(("present",)),
    )

    with pytest.raises(RuntimeError, match="LoRA keys disappeared"):
        shard_runtime._load_lora_weights_with_torch(
            Path("lora.safetensors"),
            ("missing",),
        )


def test_selected_window_blocks_filters_cache_hit_middle_blocks() -> None:
    window = ResidencyWindow(
        index=7,
        block_indices=(54, 55, 56, 57, 58, 59),
        shards=("7.safetensors",),
    )

    assert shard_runtime.selected_window_blocks(window, None) == (
        54,
        55,
        56,
        57,
        58,
        59,
    )
    assert shard_runtime.selected_window_blocks(window, lambda index: index > 57) == (
        58,
        59,
    )
    assert shard_runtime.selected_window_blocks(window, lambda _index: False) == ()


def test_lora_tensor_cache_reuses_loaded_window_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[tuple[str, ...]] = []
    source = shard_runtime.LoRASource(
        path=Path("lora.safetensors"),
        scale=1.0,
        key_plan=object(),
    )
    cache = shard_runtime.LoRAWeightWindowCache(max_windows=8)

    monkeypatch.setattr(
        shard_runtime,
        "select_lora_keys",
        lambda _plan, block_indices: tuple(f"block.{index}" for index in block_indices),
    )

    def fake_load(_path: Path, selected_keys: tuple[str, ...]) -> dict[str, str]:
        load_calls.append(selected_keys)
        return {key: f"weight:{key}" for key in selected_keys}

    monkeypatch.setattr(shard_runtime, "_load_lora_weights", fake_load)

    first = cache.load(source, (0, 1))
    second = cache.load(source, (0, 1))
    third = cache.load(source, (2, 3))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    assert second.weights == {
        "block.0": "weight:block.0",
        "block.1": "weight:block.1",
    }
    assert load_calls == [("block.0", "block.1"), ("block.2", "block.3")]


def test_shard_runtime_reuses_patched_window_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[tuple[int, ...]] = []
    lora_calls: list[tuple[int, ...]] = []
    apply_indices: list[int] = []
    window_events: list[shard_runtime.ResidencyWindowResult] = []

    def fake_load(_layout: object, indices: tuple[int, ...]) -> tuple[object, ...]:
        load_calls.append(indices)
        return tuple(
            SimpleNamespace(block_index=index, module=SimpleNamespace(name=f"block-{index}"))
            for index in indices
        )

    def fake_apply_window_loras(
        blocks: tuple[object, ...],
        *_args: object,
        **_kwargs: object,
    ) -> tuple[shard_runtime.LoRAWindowResult, ...]:
        block_indices = tuple(block.block_index for block in blocks)
        lora_calls.append(block_indices)
        return (
            shard_runtime.LoRAWindowResult(
                path="lora.safetensors",
                selected_keys=len(block_indices),
                matched_keys=len(block_indices),
                applied_layers=len(block_indices),
                weight_cache_hit=False,
            ),
        )

    monkeypatch.setattr(shard_runtime, "load_block_window", fake_load)
    monkeypatch.setattr(shard_runtime, "apply_window_loras", fake_apply_window_loras)

    runtime = shard_runtime.ShardTransformerRuntime(
        layout=SimpleNamespace(plans=(None, None)),
        windows=(
            ResidencyWindow(
                index=0,
                block_indices=(0, 1),
                shards=("0.safetensors",),
            ),
        ),
        lora_sources=(
            shard_runtime.LoRASource(
                path=Path("lora.safetensors"),
                scale=1.0,
                key_plan=object(),
            ),
        ),
        lora_targets=[],
        patched_window_cache_max_windows=1,
    )

    def apply_block(**kwargs):  # noqa: ANN003
        apply_indices.append(kwargs["idx"])
        hidden = kwargs["hidden_states"]
        encoder = kwargs["encoder_hidden_states"]
        return encoder, hidden

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
    hidden = mx.zeros((1, 2, 4))
    encoder = mx.zeros((1, 2, 4))

    first = runtime(
        transformer,
        t=0,
        config=SimpleNamespace(),
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        encoder_hidden_states_mask=mx.ones((1, 2)),
        cond_image_grid=None,
        apply_block=apply_block,
        on_window=window_events.append,
    )
    mx.eval(first)

    second = runtime(
        transformer,
        t=0,
        config=SimpleNamespace(),
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        encoder_hidden_states_mask=mx.ones((1, 2)),
        cond_image_grid=None,
        apply_block=apply_block,
        should_load_block=lambda index: index == 1,
        on_window=window_events.append,
    )
    mx.eval(second)

    assert load_calls == [(0, 1)]
    assert lora_calls == [(0, 1)]
    assert apply_indices == [0, 1, 1]
    assert [event.patched_window_cache_hit for event in window_events] == [False, True]
    assert [event.patched_window_cache_size for event in window_events] == [1, 1]


def test_shard_runtime_prepares_kquant_img_ff_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []
    window_events: list[shard_runtime.ResidencyWindowResult] = []

    monkeypatch.setattr(
        shard_runtime,
        "load_block_window",
        lambda _layout, indices: tuple(
            SimpleNamespace(block_index=index, module=SimpleNamespace(img_ff=None))
            for index in indices
        ),
    )
    monkeypatch.setattr(shard_runtime, "apply_window_loras", lambda *_a, **_k: ())
    monkeypatch.setattr(
        shard_runtime,
        "kquant_cache_bytes",
        lambda cache: 1234 if cache else 0,
    )

    def fake_prepare(blocks, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(block.block_index for block in blocks))
        cache = kwargs["cache"]
        for block in blocks:
            cache[(block.block_index, kwargs["codec"])] = object()
        return tuple(
            SimpleNamespace(cache_hit=False, block_index=block.block_index)
            for block in blocks
        )

    monkeypatch.setattr(shard_runtime, "prepare_kquant_img_ff_window", fake_prepare)

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
        kquant_img_ff=True,
        kquant_img_ff_cache_max_blocks=60,
        kquant_img_ff_codec="q6_k",
    )

    def apply_block(**kwargs):  # noqa: ANN003
        hidden = kwargs["hidden_states"]
        encoder = kwargs["encoder_hidden_states"]
        return encoder, hidden

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
    hidden = mx.zeros((1, 2, 4))
    encoder = mx.zeros((1, 2, 4))

    out = runtime(
        transformer,
        t=0,
        config=SimpleNamespace(),
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        encoder_hidden_states_mask=mx.ones((1, 2)),
        cond_image_grid=None,
        apply_block=apply_block,
        on_window=window_events.append,
    )
    mx.eval(out)

    assert calls == [(0, 1)]
    assert window_events[0].kquant_img_ff_cache_hits == 0
    assert window_events[0].kquant_img_ff_cache_misses == 2
    assert window_events[0].kquant_img_ff_cache_size == 2
    assert window_events[0].kquant_img_ff_cache_bytes == 1234
