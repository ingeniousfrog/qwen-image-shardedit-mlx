"""Windowed q6 Transformer execution with matching per-window Qwen LoRA weights."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
import gc
from pathlib import Path
import time
from typing import Any, cast

import mlx.core as mx
from safetensors import safe_open

from mflux.models.common.lora.mapping.lora_loader import LoRALoader, PatternMatch
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping

from shardedit_mlx.lora_plan import LoRAKeyPlan, plan_qwen_lora_keys, select_lora_keys
from shardedit_mlx.qwen_block_loader import (
    LoadedBlock,
    TransformerLayout,
    load_block_window,
    load_transformer_layout,
)
from shardedit_mlx.dense_img_ff_window import prepare_dense_img_ff_window
from shardedit_mlx.kquant_img_ff_window import (
    KQuantFeedForward,
    kquant_cache_bytes,
    prepare_kquant_img_ff_window,
)
from shardedit_mlx.residency_plan import (
    ResidencyWindow,
    fixed_block_windows,
    shard_block_windows,
)


BlockApply = Callable[..., tuple[mx.array, mx.array]]
BlockSelector = Callable[[int], bool]


@dataclass(frozen=True)
class LoRASource:
    path: Path
    scale: float
    key_plan: LoRAKeyPlan


@dataclass(frozen=True)
class LoRAWindowResult:
    path: str
    selected_keys: int
    matched_keys: int
    applied_layers: int
    weight_cache_hit: bool = False


@dataclass(frozen=True)
class ResidencyWindowResult:
    window_index: int
    block_indices: tuple[int, ...]
    shards: tuple[str, ...]
    load_seconds: float
    lora_seconds: float
    prepare_seconds: float
    compute_seconds: float
    release_seconds: float
    lora_selected_keys: int
    lora_matched_keys: int
    lora_applied_layers: int
    active_after_compute_gib: float
    active_after_release_gib: float
    peak_gib: float
    release_policy: str = "window"
    lora_weight_cache_hits: int = 0
    lora_weight_cache_size: int = 0
    patched_window_cache_hit: bool = False
    patched_window_cache_size: int = 0
    kquant_img_ff_cache_hits: int = 0
    kquant_img_ff_cache_misses: int = 0
    kquant_img_ff_cache_size: int = 0
    kquant_img_ff_cache_bytes: int = 0


@dataclass(frozen=True)
class LoRAWindowWeights:
    weights: dict[str, mx.array]
    cache_hit: bool


@dataclass(frozen=True)
class PatchedWindowEntry:
    blocks: tuple[LoadedBlock, ...]
    lora_results: tuple[LoRAWindowResult, ...]


class LoRAWeightWindowCache:
    """FIFO cache for already materialized per-window LoRA tensors."""

    def __init__(self, *, max_windows: int) -> None:
        if max_windows < 1:
            raise ValueError("max_windows must be >= 1")
        self.max_windows = max_windows
        self._entries: OrderedDict[
            tuple[Path, tuple[str, ...]], dict[str, mx.array]
        ] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._entries)

    def load(
        self,
        source: LoRASource,
        block_indices: tuple[int, ...],
    ) -> LoRAWindowWeights:
        selected_keys = select_lora_keys(source.key_plan, block_indices)
        key = (source.path, selected_keys)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return LoRAWindowWeights(weights=dict(cached), cache_hit=True)
        weights = _load_lora_weights(source.path, selected_keys)
        self._entries[key] = weights
        while len(self._entries) > self.max_windows:
            self._entries.popitem(last=False)
        return LoRAWindowWeights(weights=dict(weights), cache_hit=False)


class _TransformerProxy:
    """Expose original block indices to mflux's existing LoRA path resolver."""

    def __init__(self, blocks: tuple[LoadedBlock, ...], block_count: int) -> None:
        by_index = {block.block_index: block.module for block in blocks}
        self.transformer_blocks = [by_index.get(index) for index in range(block_count)]


def selected_window_blocks(
    window: ResidencyWindow,
    should_load_block: BlockSelector | None,
) -> tuple[int, ...]:
    """Return the currently needed block indices for one residency window."""

    if should_load_block is None:
        return window.block_indices
    return tuple(
        block_index
        for block_index in window.block_indices
        if should_load_block(block_index)
    )


def prepare_lora_sources(
    lora_paths: tuple[str, ...],
    lora_scales: tuple[float, ...],
    *,
    block_count: int,
) -> tuple[LoRASource, ...]:
    if len(lora_paths) != len(lora_scales):
        raise ValueError("LoRA paths and scales must have the same length")

    sources: tuple[LoRASource, ...] = ()
    for raw_path, scale in zip(lora_paths, lora_scales, strict=True):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file does not exist: {path}")
        with safe_open(path, framework="numpy") as handle:
            keys = tuple(handle.keys())
        sources = (
            *sources,
            LoRASource(
                path=path,
                scale=scale,
                key_plan=plan_qwen_lora_keys(keys, block_count=block_count),
            ),
        )
    return sources


def _torch_tensor_to_mx(tensor: Any) -> mx.array:
    import torch

    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        return mx.array(cpu_tensor.float().numpy(), dtype=mx.bfloat16)
    return mx.array(cpu_tensor.numpy())


def _load_lora_weights_with_torch(
    path: Path,
    selected_keys: tuple[str, ...],
) -> dict[str, mx.array]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = tuple(key for key in selected_keys if key not in available)
        if missing:
            raise RuntimeError(f"LoRA keys disappeared from {path}: {missing[:5]}")
        return {
            key: _torch_tensor_to_mx(handle.get_tensor(key))
            for key in selected_keys
        }


def _load_lora_weights_with_mx_load(
    path: Path,
    selected_keys: tuple[str, ...],
) -> dict[str, mx.array]:
    loaded, _ = mx.load(str(path), return_metadata=True)
    missing = tuple(key for key in selected_keys if key not in loaded)
    if missing:
        raise RuntimeError(f"LoRA keys disappeared from {path}: {missing[:5]}")
    selected = {key: loaded[key] for key in selected_keys}
    del loaded
    gc.collect()
    return selected


def _load_lora_weights(
    path: Path,
    selected_keys: tuple[str, ...],
) -> dict[str, mx.array]:
    if not selected_keys:
        return {}
    try:
        return _load_lora_weights_with_torch(path, selected_keys)
    except ImportError:
        return _load_lora_weights_with_mx_load(path, selected_keys)


def _window_weights(
    source: LoRASource,
    block_indices: tuple[int, ...],
    *,
    weight_cache: LoRAWeightWindowCache | None = None,
) -> LoRAWindowWeights:
    if weight_cache is not None:
        return weight_cache.load(source, block_indices)
    selected_keys = select_lora_keys(source.key_plan, block_indices)
    return LoRAWindowWeights(
        weights=_load_lora_weights(source.path, selected_keys),
        cache_hit=False,
    )


def apply_window_loras(
    blocks: tuple[LoadedBlock, ...],
    sources: tuple[LoRASource, ...],
    *,
    block_count: int,
    mappings: list[PatternMatch],
    weight_cache: LoRAWeightWindowCache | None = None,
) -> tuple[LoRAWindowResult, ...]:
    if not sources:
        return ()
    proxy = _TransformerProxy(blocks, block_count)
    block_indices = tuple(block.block_index for block in blocks)
    results: tuple[LoRAWindowResult, ...] = ()
    for source in sources:
        window_weights = _window_weights(
            source,
            block_indices,
            weight_cache=weight_cache,
        )
        weights = window_weights.weights
        if weights:
            applied_layers, matched_keys = LoRALoader._apply_lora_with_mapping(
                proxy,
                weights,
                source.scale,
                mappings,
                role=None,
            )
        else:
            applied_layers = 0
            matched_keys = set()
        selected_keys = len(weights)
        if len(matched_keys) != selected_keys:
            unmatched = tuple(sorted(set(weights) - matched_keys))
            raise RuntimeError(
                f"window LoRA did not match every selected key: {unmatched[:5]}"
            )
        if selected_keys and applied_layers == 0:
            raise RuntimeError(
                f"window LoRA matched keys but applied no layers: {source.path}"
            )
        results = (
            *results,
            LoRAWindowResult(
                path=str(source.path),
                selected_keys=selected_keys,
                matched_keys=len(matched_keys),
                applied_layers=applied_layers,
                weight_cache_hit=window_weights.cache_hit,
            ),
        )
        del weights
        gc.collect()
    return results


class ShardTransformerRuntime:
    def __init__(
        self,
        *,
        layout: TransformerLayout,
        windows: tuple[ResidencyWindow, ...],
        lora_sources: tuple[LoRASource, ...],
        lora_targets: list[LoRATarget],
        dense_img_ff: bool = False,
        dense_img_ff_cache_max_blocks: int = 12,
        kquant_img_ff: bool = False,
        kquant_img_ff_cache_max_blocks: int = 60,
        kquant_img_ff_codec: str = "q6_k",
        release_policy: str = "window",
        lora_tensor_cache: bool = False,
        lora_tensor_cache_max_windows: int = 8,
        patched_window_cache_max_windows: int = 0,
    ) -> None:
        if release_policy not in ("window", "step", "none", "keep-last"):
            raise ValueError(
                "release_policy must be 'window', 'step', 'none', or 'keep-last'"
            )
        if lora_tensor_cache_max_windows < 1:
            raise ValueError("lora_tensor_cache_max_windows must be >= 1")
        if patched_window_cache_max_windows < 0:
            raise ValueError("patched_window_cache_max_windows must be >= 0")
        if dense_img_ff and kquant_img_ff:
            raise ValueError("dense_img_ff and kquant_img_ff are mutually exclusive")
        if kquant_img_ff_cache_max_blocks < 1:
            raise ValueError("kquant_img_ff_cache_max_blocks must be >= 1")
        self.layout = layout
        self.windows = windows
        self.lora_sources = lora_sources
        self.lora_mappings = LoRALoader._build_pattern_mappings(lora_targets)
        self.dense_img_ff = dense_img_ff
        self.dense_img_ff_cache_max_blocks = dense_img_ff_cache_max_blocks
        self.kquant_img_ff = kquant_img_ff
        self.kquant_img_ff_cache_max_blocks = kquant_img_ff_cache_max_blocks
        self.kquant_img_ff_codec = kquant_img_ff_codec
        self.release_policy = release_policy
        self._dense_img_ff_cache: dict[int, Any] = {}
        self._kquant_img_ff_cache: dict[tuple[int, str], KQuantFeedForward] = {}
        self._lora_weight_cache = (
            LoRAWeightWindowCache(max_windows=lora_tensor_cache_max_windows)
            if lora_tensor_cache
            else None
        )
        self.patched_window_cache_max_windows = patched_window_cache_max_windows
        self._patched_window_cache: OrderedDict[
            tuple[int, ...], PatchedWindowEntry
        ] = OrderedDict()
        self._keep_last_window_blocks: tuple[LoadedBlock, ...] | None = None

    @classmethod
    def create(
        cls,
        *,
        model_path: Path,
        mode: str,
        window_size: int,
        lora_paths: tuple[str, ...],
        lora_scales: tuple[float, ...],
        dense_img_ff: bool = False,
        dense_img_ff_cache_max_blocks: int = 12,
        kquant_img_ff: bool = False,
        kquant_img_ff_cache_max_blocks: int = 60,
        kquant_img_ff_codec: str = "q6_k",
        release_policy: str = "window",
        lora_tensor_cache: bool = False,
        lora_tensor_cache_max_windows: int = 8,
        patched_window_cache_max_windows: int = 0,
    ) -> ShardTransformerRuntime:
        layout = load_transformer_layout(model_path)
        if mode == "shard":
            windows = shard_block_windows(layout.plans, layout.ordered_shards)
        elif mode == "window":
            windows = fixed_block_windows(layout.plans, window_size)
        else:
            raise ValueError(f"unsupported residency mode: {mode}")
        return cls(
            layout=layout,
            windows=windows,
            lora_sources=prepare_lora_sources(
                lora_paths,
                lora_scales,
                block_count=len(layout.plans),
            ),
            lora_targets=QwenLoRAMapping.get_mapping(),
            dense_img_ff=dense_img_ff,
            dense_img_ff_cache_max_blocks=dense_img_ff_cache_max_blocks,
            kquant_img_ff=kquant_img_ff,
            kquant_img_ff_cache_max_blocks=kquant_img_ff_cache_max_blocks,
            kquant_img_ff_codec=kquant_img_ff_codec,
            release_policy=release_policy,
            lora_tensor_cache=lora_tensor_cache,
            lora_tensor_cache_max_windows=lora_tensor_cache_max_windows,
            patched_window_cache_max_windows=patched_window_cache_max_windows,
        )

    @property
    def block_count(self) -> int:
        return len(self.layout.plans)

    @property
    def lora_key_count(self) -> int:
        return sum(source.key_plan.key_count for source in self.lora_sources)

    def detach_resident_blocks(self, transformer: Any) -> int:
        blocks = tuple(transformer.transformer_blocks)
        if len(blocks) != self.block_count:
            raise RuntimeError(
                f"expected {self.block_count} resident blocks, found {len(blocks)}"
            )
        transformer.transformer_blocks = []
        released = len(blocks)
        del blocks
        gc.collect()
        mx.clear_cache()
        return released

    def _patched_window_entry(
        self,
        block_indices: tuple[int, ...],
    ) -> PatchedWindowEntry | None:
        if self.patched_window_cache_max_windows < 1:
            return None
        entry = self._patched_window_cache.get(block_indices)
        if entry is not None:
            self._patched_window_cache.move_to_end(block_indices)
            return entry

        requested = set(block_indices)
        for cached_indices, cached_entry in reversed(self._patched_window_cache.items()):
            if not requested.issubset(cached_indices):
                continue
            by_index = {block.block_index: block for block in cached_entry.blocks}
            subset_blocks = tuple(
                by_index[index] for index in block_indices if index in by_index
            )
            if len(subset_blocks) != len(block_indices):
                continue
            self._patched_window_cache.move_to_end(cached_indices)
            return PatchedWindowEntry(
                blocks=subset_blocks,
                lora_results=cached_entry.lora_results,
            )
        return None

    def _remember_patched_window(
        self,
        block_indices: tuple[int, ...],
        entry: PatchedWindowEntry,
    ) -> None:
        if self.patched_window_cache_max_windows < 1:
            return
        self._patched_window_cache[block_indices] = entry
        self._patched_window_cache.move_to_end(block_indices)
        while len(self._patched_window_cache) > self.patched_window_cache_max_windows:
            self._patched_window_cache.popitem(last=False)

    def __call__(
        self,
        transformer: Any,
        *,
        t: int,
        config: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        cond_image_grid: Any,
        apply_block: BlockApply,
        should_load_block: BlockSelector | None = None,
        bridge_skipped_blocks: bool = False,
        on_window: Callable[[ResidencyWindowResult], None] | None = None,
    ) -> mx.array:
        hidden_states = transformer.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = transformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        encoder_hidden_states = transformer.txt_norm(encoder_hidden_states)
        encoder_hidden_states = transformer.txt_in(encoder_hidden_states)
        text_embeddings = transformer.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = transformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=transformer.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )

        last_applied_block_index: int | None = None
        window_results: list[ResidencyWindowResult] = []
        for window in self.windows:
            block_indices = selected_window_blocks(window, should_load_block)
            if not block_indices:
                continue

            if (
                bridge_skipped_blocks
                and last_applied_block_index is not None
                and block_indices[0] > last_applied_block_index + 1
            ):
                encoder_hidden_states, hidden_states = apply_block(
                    idx=last_applied_block_index + 1,
                    block=None,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    text_embeddings=text_embeddings,
                    image_rotary_embeddings=image_rotary_embeddings,
                )
                mx.eval(encoder_hidden_states, hidden_states)

            patched_entry = self._patched_window_entry(block_indices)
            patched_window_cache_hit = patched_entry is not None
            blocks: tuple[LoadedBlock, ...] = ()
            keep_blocks = patched_window_cache_hit
            load_seconds = 0.0
            lora_seconds = 0.0
            prepare_seconds = 0.0
            compute_seconds = 0.0
            lora_results: tuple[Any, ...] = ()
            kquant_cache_hits = 0
            kquant_cache_misses = 0
            active_after_compute_gib = mx.get_active_memory() / 1024**3
            active_after_release_gib = active_after_compute_gib
            release_seconds = 0.0
            try:
                if patched_entry is not None:
                    blocks = patched_entry.blocks
                    lora_results = patched_entry.lora_results
                else:
                    load_started_at = time.perf_counter()
                    blocks = load_block_window(self.layout, block_indices)
                    load_seconds = time.perf_counter() - load_started_at

                    lora_started_at = time.perf_counter()
                    lora_results = apply_window_loras(
                        blocks,
                        self.lora_sources,
                        block_count=self.block_count,
                        mappings=self.lora_mappings,
                        weight_cache=self._lora_weight_cache,
                    )
                    lora_seconds = time.perf_counter() - lora_started_at

                if self.dense_img_ff and patched_entry is None:
                    prepare_started_at = time.perf_counter()
                    prepare_dense_img_ff_window(
                        blocks,
                        cache=self._dense_img_ff_cache,
                        cache_max_blocks=self.dense_img_ff_cache_max_blocks,
                    )
                    prepare_seconds = time.perf_counter() - prepare_started_at
                if self.kquant_img_ff and patched_entry is None:
                    prepare_started_at = time.perf_counter()
                    kquant_handles = prepare_kquant_img_ff_window(
                        blocks,
                        codec=self.kquant_img_ff_codec,
                        cache=self._kquant_img_ff_cache,
                        cache_max_blocks=self.kquant_img_ff_cache_max_blocks,
                    )
                    kquant_cache_hits = sum(
                        1 for handle in kquant_handles if handle.cache_hit
                    )
                    kquant_cache_misses = sum(
                        1 for handle in kquant_handles if not handle.cache_hit
                    )
                    prepare_seconds += time.perf_counter() - prepare_started_at

                compute_started_at = time.perf_counter()
                for loaded_block in blocks:
                    encoder_hidden_states, hidden_states = apply_block(
                        idx=loaded_block.block_index,
                        block=loaded_block.module,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        text_embeddings=text_embeddings,
                        image_rotary_embeddings=image_rotary_embeddings,
                    )
                    last_applied_block_index = loaded_block.block_index
                mx.eval(encoder_hidden_states, hidden_states)
                compute_seconds = time.perf_counter() - compute_started_at
                active_after_compute_gib = mx.get_active_memory() / 1024**3
                del loaded_block
                if (
                    patched_entry is None
                    and self.patched_window_cache_max_windows > 0
                ):
                    self._remember_patched_window(
                        block_indices,
                        PatchedWindowEntry(
                            blocks=blocks,
                            lora_results=cast(
                                tuple[LoRAWindowResult, ...],
                                lora_results,
                            ),
                        ),
                    )
                    keep_blocks = True
            finally:
                if not keep_blocks:
                    if self.release_policy == "keep-last":
                        self._keep_last_window_blocks = blocks
                    else:
                        del blocks
                    if self.release_policy == "window":
                        release_started_at = time.perf_counter()
                        gc.collect()
                        mx.clear_cache()
                        release_seconds = time.perf_counter() - release_started_at
                    active_after_release_gib = mx.get_active_memory() / 1024**3
                else:
                    active_after_release_gib = mx.get_active_memory() / 1024**3

            result = ResidencyWindowResult(
                window_index=window.index,
                block_indices=block_indices,
                shards=window.shards,
                load_seconds=load_seconds,
                lora_seconds=lora_seconds,
                prepare_seconds=prepare_seconds,
                compute_seconds=compute_seconds,
                release_seconds=release_seconds,
                lora_selected_keys=sum(item.selected_keys for item in lora_results),
                lora_matched_keys=sum(item.matched_keys for item in lora_results),
                lora_applied_layers=sum(item.applied_layers for item in lora_results),
                active_after_compute_gib=active_after_compute_gib,
                active_after_release_gib=active_after_release_gib,
                peak_gib=mx.get_peak_memory() / 1024**3,
                release_policy=self.release_policy,
                lora_weight_cache_hits=sum(
                    1 for item in lora_results if item.weight_cache_hit
                ),
                lora_weight_cache_size=(
                    self._lora_weight_cache.size
                    if self._lora_weight_cache is not None
                    else 0
                ),
                patched_window_cache_hit=patched_window_cache_hit,
                patched_window_cache_size=len(self._patched_window_cache),
                kquant_img_ff_cache_hits=kquant_cache_hits,
                kquant_img_ff_cache_misses=kquant_cache_misses,
                kquant_img_ff_cache_size=len(self._kquant_img_ff_cache),
                kquant_img_ff_cache_bytes=kquant_cache_bytes(self._kquant_img_ff_cache),
            )
            window_results.append(result)

        hidden_states = transformer.norm_out(hidden_states, text_embeddings)
        output = transformer.proj_out(hidden_states)

        if self.release_policy == "step" and window_results:
            release_started_at = time.perf_counter()
            gc.collect()
            mx.clear_cache()
            release_seconds = time.perf_counter() - release_started_at
            last_result = window_results[-1]
            window_results[-1] = replace(
                last_result,
                release_seconds=last_result.release_seconds + release_seconds,
                active_after_release_gib=mx.get_active_memory() / 1024**3,
                peak_gib=mx.get_peak_memory() / 1024**3,
            )

        if on_window is not None:
            for result in window_results:
                on_window(result)
        return output
