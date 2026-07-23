"""Qwen Transformer q6 linear attribution helpers.

The runtime integration wraps MLX ``QuantizedLinear.__call__`` only when the
diagnostic flag is enabled. This module stays MLX-free so unit tests can run in
headless environments without a Metal device.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Iterator
import weakref


@dataclass(frozen=True)
class LinearCallSite:
    block: int
    site: str
    category: str


@dataclass(frozen=True)
class _RegisteredCallSite:
    module_ref: Any
    site: LinearCallSite


@dataclass(frozen=True)
class Q6LinearProfileEvent:
    step: int
    cache_hit: bool | None
    cache_reason: str | None
    category: str
    site: str
    seconds: float
    call_count: int
    blocks: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    bits: int | None
    group_size: int | None
    mode: str | None

    def details(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "cache_hit": self.cache_hit,
            "cache_reason": self.cache_reason,
            "category": self.category,
            "site": self.site,
            "call_count": self.call_count,
            "blocks": self.blocks,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class Q6LinearProfileSummary:
    category: str
    site: str
    total_seconds: float
    call_count: int
    steps: tuple[int, ...]
    blocks: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    bits: int | None
    group_size: int | None
    mode: str | None

    @property
    def mean_seconds_per_step(self) -> float:
        if not self.steps:
            return 0.0
        return self.total_seconds / len(self.steps)


@dataclass(frozen=True)
class _ProfileKey:
    step: int
    category: str
    site: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    bits: int | None
    group_size: int | None
    mode: str | None


@dataclass(frozen=True)
class _ProfileStats:
    seconds: float
    call_count: int
    blocks: tuple[int, ...]

    def add(self, *, block: int, seconds: float) -> _ProfileStats:
        return _ProfileStats(
            seconds=self.seconds + seconds,
            call_count=self.call_count + 1,
            blocks=(*self.blocks, block),
        )


@dataclass(frozen=True)
class _SummaryKey:
    category: str
    site: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    bits: int | None
    group_size: int | None
    mode: str | None


@dataclass(frozen=True)
class _SummaryStats:
    seconds: float
    call_count: int
    steps: tuple[int, ...]
    blocks: tuple[int, ...]

    def add(
        self,
        *,
        seconds: float,
        call_count: int,
        step: int,
        blocks: tuple[int, ...],
    ) -> _SummaryStats:
        return _SummaryStats(
            seconds=self.seconds + seconds,
            call_count=self.call_count + call_count,
            steps=(*self.steps, step),
            blocks=(*self.blocks, *blocks),
        )


_QWEN_BLOCK_LINEAR_SITES: tuple[tuple[str, str], ...] = (
    ("img_mod_linear", "img_mod"),
    ("txt_mod_linear", "txt_mod"),
    ("attn.to_q", "img_attn_qkv"),
    ("attn.to_k", "img_attn_qkv"),
    ("attn.to_v", "img_attn_qkv"),
    ("attn.add_q_proj", "txt_attn_qkv"),
    ("attn.add_k_proj", "txt_attn_qkv"),
    ("attn.add_v_proj", "txt_attn_qkv"),
    ("attn.attn_to_out.0", "img_attn_out"),
    ("attn.to_add_out", "txt_attn_out"),
    ("img_ff.mlp_in", "img_mlp"),
    ("img_ff.mlp_out", "img_mlp"),
    ("txt_ff.mlp_in", "txt_mlp"),
    ("txt_ff.mlp_out", "txt_mlp"),
)

_CATEGORY_ORDER = {
    "img_mod": 0,
    "txt_mod": 1,
    "img_attn_qkv": 2,
    "txt_attn_qkv": 3,
    "img_attn_out": 4,
    "txt_attn_out": 5,
    "img_mlp": 6,
    "txt_mlp": 7,
}


def qwen_block_linear_call_sites(block: Any, block_index: int) -> dict[int, LinearCallSite]:
    """Return known Qwen Transformer linear modules for one zero-based block."""

    sites: dict[int, LinearCallSite] = {}
    for path, category in _QWEN_BLOCK_LINEAR_SITES:
        module = _resolve_path(block, path)
        if module is None:
            continue
        sites[id(module)] = LinearCallSite(
            block=block_index + 1,
            site=path,
            category=category,
        )
    return sites


class Q6LinearProfiler:
    def __init__(self) -> None:
        self._sites_by_module_id: dict[int, _RegisteredCallSite] = {}
        self._records: dict[_ProfileKey, _ProfileStats] = {}
        self._current_step: int | None = None
        self._current_block: int | None = None

    def register_block(self, block: Any, block_index: int) -> None:
        for module_id, site in qwen_block_linear_call_sites(block, block_index).items():
            module = _resolve_path(block, site.site)
            if module is None:
                continue
            for profile_module in _profile_modules_for_linear(module):
                self._sites_by_module_id[id(profile_module)] = _RegisteredCallSite(
                    module_ref=_module_ref(profile_module),
                    site=site,
                )

    def should_record(self, module: Any) -> bool:
        return (
            self._current_step is not None
            and self._current_block is not None
            and self._site_for_module(module) is not None
        )

    def _site_for_module(self, module: Any) -> LinearCallSite | None:
        registered = self._sites_by_module_id.get(id(module))
        if registered is None:
            return None
        if registered.module_ref() is not module:
            return None
        return registered.site

    @contextmanager
    def block_context(self, *, step: int | None, block: int | None) -> Iterator[None]:
        previous_step = self._current_step
        previous_block = self._current_block
        self._current_step = step
        self._current_block = block
        try:
            yield
        finally:
            self._current_step = previous_step
            self._current_block = previous_block

    def record_call(
        self,
        module: Any,
        *,
        input_value: Any,
        output_value: Any,
        seconds: float,
    ) -> None:
        if self._current_step is None or self._current_block is None:
            return
        site = self._site_for_module(module)
        if site is None:
            return
        if seconds < 0.0:
            return
        key = _ProfileKey(
            step=self._current_step,
            category=site.category,
            site=site.site,
            input_shape=_shape_tuple(input_value),
            output_shape=_shape_tuple(output_value),
            bits=_optional_int(getattr(module, "bits", None)),
            group_size=_optional_int(getattr(module, "group_size", None)),
            mode=_optional_str(getattr(module, "mode", None)),
        )
        existing = self._records.get(key, _ProfileStats(seconds=0.0, call_count=0, blocks=()))
        self._records[key] = existing.add(block=site.block, seconds=seconds)

    def drain_step(
        self,
        *,
        step: int,
        cache_hit: bool | None,
        cache_reason: str | None,
    ) -> tuple[Q6LinearProfileEvent, ...]:
        selected = tuple(
            (key, stats)
            for key, stats in self._records.items()
            if key.step == step
        )
        self._records = {
            key: stats
            for key, stats in self._records.items()
            if key.step != step
        }
        return tuple(
            Q6LinearProfileEvent(
                step=key.step,
                cache_hit=cache_hit,
                cache_reason=cache_reason,
                category=key.category,
                site=key.site,
                seconds=stats.seconds,
                call_count=stats.call_count,
                blocks=_compact_blocks(stats.blocks),
                input_shape=key.input_shape,
                output_shape=key.output_shape,
                bits=key.bits,
                group_size=key.group_size,
                mode=key.mode,
            )
            for key, stats in sorted(selected, key=_sort_profile_item)
        )


def summarize_q6_linear_events(
    events: Sequence[Mapping[str, Any]],
    *,
    full_miss_only: bool = True,
) -> tuple[Q6LinearProfileSummary, ...]:
    """Aggregate emitted q6 linear profile events by callsite and shape."""

    grouped: dict[_SummaryKey, _SummaryStats] = {}
    for event in events:
        if event.get("name") != "q6_linear_profile":
            continue
        if full_miss_only and event.get("cache_hit") is True:
            continue
        seconds = _optional_float(event.get("seconds"))
        if seconds is None:
            continue
        step = _optional_int(event.get("step"))
        if step is None:
            continue
        key = _SummaryKey(
            category=str(event.get("category") or "unknown"),
            site=str(event.get("site") or "unknown"),
            input_shape=_shape_tuple_from_sequence(event.get("input_shape")),
            output_shape=_shape_tuple_from_sequence(event.get("output_shape")),
            bits=_optional_int(event.get("bits")),
            group_size=_optional_int(event.get("group_size")),
            mode=_optional_str(event.get("mode")),
        )
        existing = grouped.get(
            key,
            _SummaryStats(seconds=0.0, call_count=0, steps=(), blocks=()),
        )
        grouped[key] = existing.add(
            seconds=seconds,
            call_count=_optional_int(event.get("call_count")) or 0,
            step=step,
            blocks=_parse_blocks(str(event.get("blocks") or "")),
        )
    return tuple(
        Q6LinearProfileSummary(
            category=key.category,
            site=key.site,
            total_seconds=stats.seconds,
            call_count=stats.call_count,
            steps=tuple(sorted(set(stats.steps))),
            blocks=_compact_blocks(stats.blocks),
            input_shape=key.input_shape,
            output_shape=key.output_shape,
            bits=key.bits,
            group_size=key.group_size,
            mode=key.mode,
        )
        for key, stats in sorted(
            grouped.items(),
            key=lambda item: (
                -item[1].seconds,
                _CATEGORY_ORDER.get(item[0].category, 99),
                item[0].site,
            ),
        )
    )


def _resolve_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            current = getattr(current, part, None)
            if current is None:
                return None
    return current


def _module_ref(module: Any) -> Any:
    try:
        return weakref.ref(module)
    except TypeError:
        return lambda: module


def _profile_modules_for_linear(module: Any) -> tuple[Any, ...]:
    """Return the callable linear modules that can reach QuantizedLinear.__call__."""

    modules: tuple[Any, ...] = (module,)
    for attr in ("linear", "base_linear"):
        nested = getattr(module, attr, None)
        if nested is not None and not any(existing is nested for existing in modules):
            modules = (*modules, nested)
    return modules


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(part) for part in shape)
    except (TypeError, ValueError):
        return ()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return None
    return parsed


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _shape_tuple_from_sequence(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return ()


def _parse_blocks(value: str) -> tuple[int, ...]:
    blocks: tuple[int, ...] = ()
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        if "-" in stripped:
            start_raw, end_raw = stripped.split("-", 1)
            start = _optional_int(start_raw)
            end = _optional_int(end_raw)
            if start is None or end is None or end < start:
                continue
            blocks = (*blocks, *tuple(range(start, end + 1)))
            continue
        block = _optional_int(stripped)
        if block is not None:
            blocks = (*blocks, block)
    return blocks


def _compact_blocks(blocks: tuple[int, ...]) -> str:
    unique = tuple(sorted(set(blocks)))
    if not unique:
        return ""
    ranges: list[str] = []
    start = unique[0]
    previous = unique[0]
    for block in unique[1:]:
        if block == previous + 1:
            previous = block
            continue
        ranges.append(_format_range(start, previous))
        start = block
        previous = block
    ranges.append(_format_range(start, previous))
    return ",".join(ranges)


def _format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _sort_profile_item(item: tuple[_ProfileKey, _ProfileStats]) -> tuple[Any, ...]:
    key, stats = item
    blocks = tuple(sorted(set(stats.blocks)))
    first_block = blocks[0] if blocks else 0
    return (
        key.step,
        _CATEGORY_ORDER.get(key.category, 99),
        key.site,
        key.input_shape,
        key.output_shape,
        first_block,
    )
