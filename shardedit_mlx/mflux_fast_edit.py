"""mflux-compatible Qwen Image Edit entry point with exact CFG pruning."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardedit_mlx.bridge_error_correlation import (
    correlate_bridge_error_with_uniqueness,
    even_index_values,
    uniqueness_from_similarities,
)
from shardedit_mlx.condition_token_merge import (
    ConditionGrid,
    ConditionMergePlan,
    TextMergePlan,
    build_condition_merge_plan,
    build_text_merge_plan,
    should_merge_condition_block,
)
from shardedit_mlx.q6_linear_profile import Q6LinearProfiler
from shardedit_mlx.selective_refill import (
    DEFAULT_SELECTIVE_REFILL_MODE,
    RESIDUAL_ADJUST_MODES,
    SELECTIVE_REFILL_MODES,
    SUBSET_REFILL_MODES,
    build_image_gather_indices,
    select_unique_even_indices,
    should_apply_selective_refill,
    uniqueness_scaled_residual_scales,
)
from shardedit_mlx.token_redundancy import redundancy_summary
from shardedit_mlx.token_redundancy_heatmap import similarities_to_grid, similarity_to_rgb


GuidanceFunction = Callable[[Any, Any, float], Any]
DEFAULT_RESIDENCY_MODE = "shard"
DEFAULT_RESIDENCY_WINDOW_SIZE = 8
DEFAULT_RELEASE_POLICY = "window"
DEFAULT_DENSE_IMG_FF_CACHE_MAX_BLOCKS = 60
DEFAULT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS = 60
DEFAULT_KQUANT_IMG_FF_CODEC = "q6_k"
DEFAULT_LORA_TENSOR_CACHE_MAX_WINDOWS = 8
DEFAULT_CACHE_ANCHOR_MODE = "residual"
DEFAULT_CACHE_PREDICTOR = "last"
DEFAULT_CACHE_THRESHOLD_SCHEDULE = "fixed"
DEFAULT_CACHE_REGION_POLICY = "all"
DEFAULT_REFERENCE_CONDITIONING_SIZE = "upstream"
DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE = 512
DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH = 576
DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT = 768
DEFAULT_SELECTIVE_REFILL_DAMPEN = 1.0
DEFAULT_CONDITION_TOKEN_MERGE_STRIDE = 2
DEFAULT_CONDITION_TOKEN_MERGE_START_BLOCK = 2
DEFAULT_CONDITION_TOKEN_MERGE_BACK_BLOCKS = 2
DEFAULT_TEXT_TOKEN_MERGE_STRIDE = 2
DEFAULT_TEXT_TOKEN_MERGE_START_BLOCK = 2
DEFAULT_TEXT_TOKEN_MERGE_BACK_BLOCKS = 2
CACHE_PREDICTORS = (
    "last",
    "linear",
    "linear-residual",
    "quadratic",
    "quadratic-residual",
    "adams-bashforth",
    "adams-bashforth-residual",
)
CACHE_THRESHOLD_SCHEDULES = ("fixed", "sigma", "flow-aware", "flow-aware-veto")
CACHE_REGION_POLICIES = ("all", "target-conservative", "condition-conservative")
REFERENCE_CONDITIONING_SIZE_POLICIES = (
    "upstream",
    "original",
    "short-side",
    "short-side-512",
    "fit-box",
)
RELEASE_POLICIES = ("window", "step", "none", "keep-last")


@dataclass(frozen=True)
class FileCacheSignature:
    """Stable-enough local file identity for in-process preprocessing caches."""

    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class PromptEncodingCacheKey:
    prompt: str
    negative_prompt: str | None
    image_signatures: tuple[FileCacheSignature, ...]
    vl_width: int | None
    vl_height: int | None
    guidance: float
    use_picture_prefix: bool


@dataclass(frozen=True)
class ConditioningLatentsCacheKey:
    image_signatures: tuple[FileCacheSignature, ...]
    height: int
    width: int
    vl_width: int | None
    vl_height: int | None


@dataclass(frozen=True)
class RuntimeOptions:
    """qwen-image-shardedit-mlx-only options removed before invoking the upstream CLI."""

    eval_every_n_blocks: int = 0
    probe_blocks: tuple[int, ...] = ()
    token_redundancy_blocks: tuple[int, ...] = ()
    token_redundancy_heatmap_dir: Path | None = None
    bridge_error_diagnose: bool = False
    bridge_error_heatmap_dir: Path | None = None
    selective_refill_fraction: float = 0.0
    selective_refill_mode: str = DEFAULT_SELECTIVE_REFILL_MODE
    selective_refill_dampen: float = DEFAULT_SELECTIVE_REFILL_DAMPEN
    selective_refill_min_step: int = 0
    cache_threshold: float = 0.0
    cache_max_consecutive: int = 1
    cache_warmup_steps: int = 1
    cache_back_blocks: int = 0
    cache_anchor_mode: str = DEFAULT_CACHE_ANCHOR_MODE
    cache_predictor: str = DEFAULT_CACHE_PREDICTOR
    cache_threshold_schedule: str = DEFAULT_CACHE_THRESHOLD_SCHEDULE
    cache_region_policy: str = DEFAULT_CACHE_REGION_POLICY
    reference_conditioning_size: str = DEFAULT_REFERENCE_CONDITIONING_SIZE
    reference_conditioning_short_side: int = DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE
    reference_conditioning_max_width: int = DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH
    reference_conditioning_max_height: int = DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT
    residency_mode: str = DEFAULT_RESIDENCY_MODE
    residency_window_size: int = DEFAULT_RESIDENCY_WINDOW_SIZE
    release_policy: str = DEFAULT_RELEASE_POLICY
    dense_img_ff_window: bool = False
    dense_img_ff_cache_max_blocks: int = DEFAULT_DENSE_IMG_FF_CACHE_MAX_BLOCKS
    kquant_img_ff_window: bool = False
    kquant_img_ff_cache_max_blocks: int = DEFAULT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS
    kquant_img_ff_codec: str = DEFAULT_KQUANT_IMG_FF_CODEC
    lora_tensor_cache: bool = False
    lora_tensor_cache_max_windows: int = DEFAULT_LORA_TENSOR_CACHE_MAX_WINDOWS
    patched_window_cache_max_windows: int = 0
    condition_token_merge: bool = False
    condition_token_merge_stride: int = DEFAULT_CONDITION_TOKEN_MERGE_STRIDE
    condition_token_merge_start_block: int = DEFAULT_CONDITION_TOKEN_MERGE_START_BLOCK
    condition_token_merge_back_blocks: int = DEFAULT_CONDITION_TOKEN_MERGE_BACK_BLOCKS
    text_token_merge: bool = False
    text_token_merge_stride: int = DEFAULT_TEXT_TOKEN_MERGE_STRIDE
    text_token_merge_start_block: int = DEFAULT_TEXT_TOKEN_MERGE_START_BLOCK
    text_token_merge_back_blocks: int = DEFAULT_TEXT_TOKEN_MERGE_BACK_BLOCKS
    q6_linear_profile: bool = False
    profile: bool = False


@dataclass(frozen=True)
class TransformerCallState:
    """Tracks paired positive/negative Transformer calls for one denoise step."""

    key: tuple[int, int] | None = None
    calls_for_key: int = 0


@dataclass(frozen=True)
class ResidualCacheState:
    """Tracks the last full-compute anchor and consecutive cache hits."""

    has_anchor: bool = False
    consecutive_hits: int = 0


@dataclass(frozen=True)
class AnchorPrediction:
    value: Any
    scale: float | None
    order: int
    method: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CacheThresholdAdjustment:
    value: float
    progress: float | None
    coordinate: str
    cosine_factor: float | None = None
    magnitude_factor: float | None = None
    history_factor: float | None = None
    veto_threshold: float | None = None


def decide_transformer_call(
    state: TransformerCallState,
    key: tuple[int, int],
    unit_guidance: bool,
) -> tuple[bool, TransformerCallState]:
    """Run only the first Transformer call for a unit-guidance step."""

    if not unit_guidance:
        return True, TransformerCallState()
    calls_for_key = state.calls_for_key + 1 if state.key == key else 1
    next_state = TransformerCallState(key=key, calls_for_key=calls_for_key)
    return calls_for_key == 1, next_state


def decide_residual_cache(
    state: ResidualCacheState,
    *,
    step: int,
    warmup_steps: int,
    threshold: float,
    max_consecutive: int,
    relative_l1: float | None,
) -> tuple[bool, ResidualCacheState, str]:
    """Choose a cache hit while keeping periodic full-compute anchors."""

    if threshold <= 0.0:
        return False, state, "disabled"
    if not state.has_anchor or step <= warmup_steps:
        return False, ResidualCacheState(has_anchor=True), "warmup"
    if state.consecutive_hits >= max_consecutive:
        return False, ResidualCacheState(has_anchor=True), "max_consecutive"
    if relative_l1 is None:
        raise ValueError("relative_l1 is required for a residual cache decision")
    if relative_l1 < threshold:
        return (
            True,
            ResidualCacheState(
                has_anchor=True,
                consecutive_hits=state.consecutive_hits + 1,
            ),
            "diff_hit",
        )
    return False, ResidualCacheState(has_anchor=True), "diff_miss"


def scheduled_cache_threshold(
    base_threshold: float,
    schedule: str,
    *,
    step: int,
    total_steps: int,
    current_sigma: float | None = None,
    first_sigma: float | None = None,
    final_sigma: float | None = None,
) -> tuple[float, float | None, str]:
    """Return the effective cache threshold for the current flow-matching step."""

    if schedule == "fixed" or base_threshold <= 0.0:
        return base_threshold, None, "fixed"
    if schedule not in ("sigma", "flow-aware", "flow-aware-veto"):
        raise ValueError(f"unsupported cache threshold schedule: {schedule}")

    progress: float | None = None
    if (
        current_sigma is not None
        and first_sigma is not None
        and final_sigma is not None
        and math.isfinite(current_sigma)
        and math.isfinite(first_sigma)
        and math.isfinite(final_sigma)
        and not math.isclose(first_sigma, final_sigma)
    ):
        sigma_position = (current_sigma - final_sigma) / (first_sigma - final_sigma)
        progress = 1.0 - max(0.0, min(1.0, sigma_position))
        coordinate = "sigma"
    else:
        denominator = max(total_steps - 1, 1)
        progress = max(0.0, min(1.0, (step - 1) / denominator))
        coordinate = "step"

    scale = 0.65 + 0.35 * progress
    return base_threshold * scale, progress, coordinate


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def flow_aware_cache_threshold(
    base_threshold: float,
    *,
    step: int,
    total_steps: int,
    current_sigma: float | None = None,
    first_sigma: float | None = None,
    final_sigma: float | None = None,
    prediction_cosine: float | None = None,
    magnitude_ratio: float | None = None,
    history_relative_l1: float | None = None,
) -> CacheThresholdAdjustment:
    """Apply TeaCache/OriCache-style flow signals to the sigma threshold."""

    threshold, progress, coordinate = scheduled_cache_threshold(
        base_threshold,
        "sigma",
        step=step,
        total_steps=total_steps,
        current_sigma=current_sigma,
        first_sigma=first_sigma,
        final_sigma=final_sigma,
    )
    cosine_factor: float | None = None
    magnitude_factor: float | None = None
    history_factor: float | None = None
    if base_threshold <= 0.0:
        return CacheThresholdAdjustment(threshold, progress, coordinate)

    if prediction_cosine is not None and math.isfinite(prediction_cosine):
        cosine_factor = 0.55 + 0.45 * _clamp_unit((prediction_cosine - 0.9) / 0.1)
        threshold *= cosine_factor
    if magnitude_ratio is not None and math.isfinite(magnitude_ratio) and magnitude_ratio > 0.0:
        magnitude_delta = abs(math.log(magnitude_ratio))
        magnitude_factor = max(0.55, 1.0 - min(0.45, magnitude_delta * 0.75))
        threshold *= magnitude_factor
    if history_relative_l1 is not None and math.isfinite(history_relative_l1):
        excess_error = max(0.0, history_relative_l1 - 0.25)
        history_factor = max(0.55, 1.0 - min(0.45, excess_error * 0.9))
        threshold *= history_factor

    return CacheThresholdAdjustment(
        value=threshold,
        progress=progress,
        coordinate=coordinate,
        cosine_factor=cosine_factor,
        magnitude_factor=magnitude_factor,
        history_factor=history_factor,
    )


def cache_threshold_adjustment(
    base_threshold: float,
    schedule: str,
    *,
    step: int,
    total_steps: int,
    current_sigma: float | None = None,
    first_sigma: float | None = None,
    final_sigma: float | None = None,
    prediction_cosine: float | None = None,
    magnitude_ratio: float | None = None,
    history_relative_l1: float | None = None,
) -> CacheThresholdAdjustment:
    """Return a threshold plus the factors used to derive it."""

    if schedule in ("flow-aware", "flow-aware-veto"):
        flow_adjustment = flow_aware_cache_threshold(
            base_threshold,
            step=step,
            total_steps=total_steps,
            current_sigma=current_sigma,
            first_sigma=first_sigma,
            final_sigma=final_sigma,
            prediction_cosine=prediction_cosine,
            magnitude_ratio=magnitude_ratio,
            history_relative_l1=history_relative_l1,
        )
        if schedule == "flow-aware":
            return flow_adjustment
        return CacheThresholdAdjustment(
            value=base_threshold,
            progress=flow_adjustment.progress,
            coordinate=flow_adjustment.coordinate,
            cosine_factor=flow_adjustment.cosine_factor,
            magnitude_factor=flow_adjustment.magnitude_factor,
            history_factor=flow_adjustment.history_factor,
            veto_threshold=flow_adjustment.value,
        )
    threshold, progress, coordinate = scheduled_cache_threshold(
        base_threshold,
        schedule,
        step=step,
        total_steps=total_steps,
        current_sigma=current_sigma,
        first_sigma=first_sigma,
        final_sigma=final_sigma,
    )
    return CacheThresholdAdjustment(threshold, progress, coordinate)


def should_flow_veto_cache_hit(
    *,
    schedule: str,
    cache_hit: bool,
    relative_l1: float | None,
    base_threshold: float,
    veto_threshold: float | None,
    prediction_cosine: float | None = None,
    magnitude_ratio: float | None = None,
) -> tuple[bool, str | None]:
    """Use flow-aware signals as a veto while preserving fixed F1B2 cadence."""

    if schedule != "flow-aware-veto" or not cache_hit:
        return False, None
    if relative_l1 is None or veto_threshold is None or base_threshold <= 0.0:
        return False, None
    if not math.isfinite(relative_l1) or not math.isfinite(veto_threshold):
        return False, None

    boundary_floor = max(veto_threshold, base_threshold * 0.75)
    if relative_l1 >= boundary_floor:
        return True, "flow_veto_boundary"

    risk_floor = max(veto_threshold, base_threshold * 0.5)
    if (
        prediction_cosine is not None
        and math.isfinite(prediction_cosine)
        and prediction_cosine < 0.9
        and relative_l1 >= risk_floor
    ):
        return True, "flow_veto_cosine"
    if magnitude_ratio is not None and math.isfinite(magnitude_ratio) and magnitude_ratio > 0.0:
        if (magnitude_ratio < 0.5 or magnitude_ratio > 2.0) and relative_l1 >= risk_floor:
            return True, "flow_veto_magnitude"
    return False, None


def linear_extrapolation_scale(
    *,
    previous_coordinate: float | None,
    anchor_coordinate: float | None,
    current_coordinate: float | None,
) -> float | None:
    """Compute a first-order Taylor-style extrapolation scale between anchors."""

    if (
        previous_coordinate is None
        or anchor_coordinate is None
        or current_coordinate is None
        or not math.isfinite(previous_coordinate)
        or not math.isfinite(anchor_coordinate)
        or not math.isfinite(current_coordinate)
    ):
        return None
    denominator = anchor_coordinate - previous_coordinate
    if math.isclose(denominator, 0.0):
        return None
    scale = (current_coordinate - anchor_coordinate) / denominator
    return max(0.0, min(2.0, scale))


def _finite_coordinate(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _normalized_predictor_name(predictor: str) -> str:
    if predictor in ("linear", "linear-residual"):
        return "linear"
    if predictor in ("quadratic", "quadratic-residual"):
        return "quadratic"
    if predictor in ("adams-bashforth", "adams-bashforth-residual"):
        return "adams-bashforth"
    if predictor == "last":
        return "last"
    raise ValueError(f"unsupported cache predictor: {predictor}")


def _last_anchor_prediction(
    cached_anchor: Any,
    *,
    requested_method: str,
    fallback_reason: str | None = None,
) -> AnchorPrediction:
    return AnchorPrediction(
        value=cached_anchor,
        scale=None,
        order=0,
        method=requested_method,
        fallback_reason=fallback_reason,
    )


def _quadratic_prediction(
    *,
    older_anchor: Any,
    previous_anchor: Any,
    cached_anchor: Any,
    older_coordinate: float,
    previous_coordinate: float,
    anchor_coordinate: float,
    current_coordinate: float,
) -> Any | None:
    coordinates = (older_coordinate, previous_coordinate, anchor_coordinate)
    values = (older_anchor, previous_anchor, cached_anchor)
    weights: list[float] = []
    for index, coordinate in enumerate(coordinates):
        numerator = 1.0
        denominator = 1.0
        for other_index, other_coordinate in enumerate(coordinates):
            if index == other_index:
                continue
            denominator *= coordinate - other_coordinate
            numerator *= current_coordinate - other_coordinate
        if math.isclose(denominator, 0.0):
            return None
        weights.append(numerator / denominator)
    return values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2]


def _adams_bashforth_prediction(
    *,
    older_anchor: Any,
    previous_anchor: Any,
    cached_anchor: Any,
    older_coordinate: float,
    previous_coordinate: float,
    anchor_coordinate: float,
    current_coordinate: float,
) -> Any | None:
    older_delta = previous_coordinate - older_coordinate
    latest_delta = anchor_coordinate - previous_coordinate
    if math.isclose(older_delta, 0.0) or math.isclose(latest_delta, 0.0):
        return None
    older_velocity = (previous_anchor - older_anchor) / older_delta
    latest_velocity = (cached_anchor - previous_anchor) / latest_delta
    current_delta = current_coordinate - anchor_coordinate
    return cached_anchor + current_delta * (1.5 * latest_velocity - 0.5 * older_velocity)


def select_predicted_anchor(
    *,
    predictor: str,
    cached_anchor: Any,
    previous_anchor: Any = None,
    older_anchor: Any = None,
    older_coordinate: float | None = None,
    previous_coordinate: float | None = None,
    anchor_coordinate: float | None = None,
    current_coordinate: float | None = None,
) -> AnchorPrediction:
    """Select the cached anchor or a Taylor-style extrapolated anchor."""

    method = _normalized_predictor_name(predictor)
    if method == "last":
        return _last_anchor_prediction(cached_anchor, requested_method=method)

    previous_coordinate = _finite_coordinate(previous_coordinate)
    anchor_coordinate = _finite_coordinate(anchor_coordinate)
    current_coordinate = _finite_coordinate(current_coordinate)
    if previous_anchor is None or previous_coordinate is None:
        return _last_anchor_prediction(
            cached_anchor,
            requested_method=method,
            fallback_reason="insufficient_history",
        )
    scale = linear_extrapolation_scale(
        previous_coordinate=previous_coordinate,
        anchor_coordinate=anchor_coordinate,
        current_coordinate=current_coordinate,
    )
    if scale is None:
        return _last_anchor_prediction(
            cached_anchor,
            requested_method=method,
            fallback_reason="invalid_coordinates",
        )
    if method == "linear":
        return AnchorPrediction(
            value=cached_anchor + (cached_anchor - previous_anchor) * scale,
            scale=scale,
            order=1,
            method=method,
        )

    older_coordinate = _finite_coordinate(older_coordinate)
    if older_anchor is None or older_coordinate is None:
        return AnchorPrediction(
            value=cached_anchor + (cached_anchor - previous_anchor) * scale,
            scale=scale,
            order=1,
            method=method,
            fallback_reason="linear_fallback",
        )
    if method == "quadratic":
        predicted = _quadratic_prediction(
            older_anchor=older_anchor,
            previous_anchor=previous_anchor,
            cached_anchor=cached_anchor,
            older_coordinate=older_coordinate,
            previous_coordinate=previous_coordinate,
            anchor_coordinate=anchor_coordinate,
            current_coordinate=current_coordinate,
        )
    else:
        predicted = _adams_bashforth_prediction(
            older_anchor=older_anchor,
            previous_anchor=previous_anchor,
            cached_anchor=cached_anchor,
            older_coordinate=older_coordinate,
            previous_coordinate=previous_coordinate,
            anchor_coordinate=anchor_coordinate,
            current_coordinate=current_coordinate,
        )
    if predicted is None:
        return AnchorPrediction(
            value=cached_anchor + (cached_anchor - previous_anchor) * scale,
            scale=scale,
            order=1,
            method=method,
            fallback_reason="linear_fallback",
        )
    return AnchorPrediction(
        value=predicted,
        scale=scale,
        order=2,
        method=method,
    )


def select_cache_decision_metric(
    *,
    policy: str,
    global_relative_l1: float | None,
    target_relative_l1: float | None = None,
    condition_relative_l1: float | None = None,
) -> float | None:
    """Apply a region policy to the cache hit/miss error signal."""

    if policy not in CACHE_REGION_POLICIES:
        raise ValueError(f"unsupported cache region policy: {policy}")
    if global_relative_l1 is None:
        return None
    if policy == "target-conservative" and target_relative_l1 is not None:
        return max(global_relative_l1, target_relative_l1)
    if policy == "condition-conservative" and condition_relative_l1 is not None:
        return max(global_relative_l1, condition_relative_l1)
    return global_relative_l1


def vae_encode_condition_dimensions(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Extract conditioning dimensions from LatentCreator.encode_image arguments."""

    height = kwargs.get("height", args[2] if len(args) > 2 else None)
    width = kwargs.get("width", args[3] if len(args) > 3 else None)
    try:
        condition_height = int(height) if height is not None else None
        condition_width = int(width) if width is not None else None
    except (TypeError, ValueError):
        return None, None
    return condition_height, condition_width


def round_dimension_to_multiple(value: float, multiple: int = 32) -> int:
    """Round image dimensions to the grid expected by Qwen conditioning paths."""

    return max(multiple, int(round(value / multiple) * multiple))


def fit_dimension_to_multiple(value: float, max_value: int, multiple: int = 32) -> int:
    """Fit one dimension to a multiple without exceeding a configured maximum."""

    if max_value < multiple:
        raise ValueError(f"reference conditioning max dimension must be >= {multiple}")
    capped = min(value, max_value)
    fitted = int(math.floor(capped / multiple) * multiple)
    return max(multiple, fitted)


def reference_conditioning_dimensions(
    *,
    policy: str,
    image_width: int,
    image_height: int,
    short_side: int = DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE,
    max_width: int = DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH,
    max_height: int = DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT,
) -> tuple[int, int] | None:
    """Return override dimensions for reference VAE conditioning, if requested."""

    if policy == "upstream":
        return None
    if image_width <= 0 or image_height <= 0:
        raise ValueError("reference image dimensions must be positive")
    if short_side <= 0:
        raise ValueError("reference conditioning short side must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("reference conditioning max dimensions must be positive")
    if policy == "original":
        return (
            round_dimension_to_multiple(image_width),
            round_dimension_to_multiple(image_height),
        )
    if policy in ("short-side", "short-side-512"):
        scale = short_side / min(image_width, image_height)
        return (
            round_dimension_to_multiple(image_width * scale),
            round_dimension_to_multiple(image_height * scale),
        )
    if policy == "fit-box":
        scale = min(max_width / image_width, max_height / image_height, 1.0)
        return (
            fit_dimension_to_multiple(image_width * scale, max_width),
            fit_dimension_to_multiple(image_height * scale, max_height),
        )
    raise ValueError(f"unsupported reference conditioning size policy: {policy}")


def replace_or_add_conditioning_size(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return arguments with vl_width/vl_height replaced by an override size."""

    if len(args) > 5:
        updated_args = list(args)
        updated_args[4] = width
        updated_args[5] = height
        return tuple(updated_args), kwargs
    updated_kwargs = dict(kwargs)
    updated_kwargs["vl_width"] = width
    updated_kwargs["vl_height"] = height
    return args, updated_kwargs


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _probe_blocks(value: str) -> tuple[int, ...]:
    try:
        blocks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of integers") from exc
    if not blocks:
        raise argparse.ArgumentTypeError("must contain at least one block")
    if any(block < 1 or block > 60 for block in blocks):
        raise argparse.ArgumentTypeError("blocks must be between 1 and 60")
    return tuple(sorted(set(blocks)))


def normalize_image_paths(image_paths: list[str] | tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(image_paths, str):
        return (image_paths,)
    return tuple(str(path) for path in image_paths)


def infer_prompt_cache_picture_prefix(
    image_paths: list[str] | tuple[str, ...] | str,
    tokenizer_use_picture_prefix: bool | None,
) -> bool:
    """Infer the prompt-cache tokenizer mode even after mflux releases the tokenizer."""

    return bool(tokenizer_use_picture_prefix) or len(normalize_image_paths(image_paths)) > 1


def file_cache_signature(path: str) -> FileCacheSignature:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return FileCacheSignature(
        path=str(resolved),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def image_cache_signatures(image_paths: list[str] | tuple[str, ...] | str) -> tuple[FileCacheSignature, ...]:
    return tuple(file_cache_signature(path) for path in normalize_image_paths(image_paths))


def prompt_encoding_cache_key(
    *,
    prompt: str,
    negative_prompt: str | None,
    image_paths: list[str] | tuple[str, ...] | str,
    vl_width: int | None,
    vl_height: int | None,
    guidance: float,
    use_picture_prefix: bool,
) -> PromptEncodingCacheKey:
    return PromptEncodingCacheKey(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_signatures=image_cache_signatures(image_paths),
        vl_width=vl_width,
        vl_height=vl_height,
        guidance=guidance,
        use_picture_prefix=use_picture_prefix,
    )


def conditioning_latents_cache_key(
    *,
    image_paths: list[str] | tuple[str, ...] | str,
    height: int,
    width: int,
    vl_width: int | None,
    vl_height: int | None,
) -> ConditioningLatentsCacheKey:
    return ConditioningLatentsCacheKey(
        image_signatures=image_cache_signatures(image_paths),
        height=height,
        width=width,
        vl_width=vl_width,
        vl_height=vl_height,
    )


def parse_runtime_options(argv: list[str]) -> tuple[RuntimeOptions, list[str]]:
    """Parse qwen-image-shardedit-mlx flags while preserving all upstream mflux arguments."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--shardedit-eval-every-n-blocks",
        type=_non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--shardedit-probe-blocks",
        type=_probe_blocks,
        default=(),
    )
    parser.add_argument(
        "--shardedit-token-redundancy-blocks",
        type=_probe_blocks,
        default=(),
        help=(
            "diagnostic only: report bipartite token similarity (mergeability) "
            "at these block boundaries; does not change any computation"
        ),
    )
    parser.add_argument(
        "--shardedit-token-redundancy-heatmap-dir",
        type=Path,
        default=None,
        help=(
            "diagnostic only: alongside --shardedit-token-redundancy-blocks, also save a "
            "PNG heatmap per (step, block) of the target region's per-token best-match "
            "similarity, reshaped back into the image's patch grid, so low-redundancy "
            "('unique') tokens can be visually correlated with detail regions "
            "(hair/lace/eyes) in the generated image; does not change any computation"
        ),
    )
    parser.add_argument(
        "--shardedit-bridge-error-diagnose",
        action="store_true",
        help=(
            "diagnostic only: on a full (no-skip) pass with --shardedit-cache-back-blocks set, "
            "compare predicted vs actual middle residual per target token and correlate "
            "with bipartite uniqueness; does not change any computation"
        ),
    )
    parser.add_argument(
        "--shardedit-bridge-error-heatmap-dir",
        type=Path,
        default=None,
        help=(
            "diagnostic only: alongside --shardedit-bridge-error-diagnose, save a PNG of "
            "per-token middle-bridge abs error (even indices) for visual comparison"
        ),
    )
    parser.add_argument(
        "--shardedit-selective-refill-fraction",
        type=_unit_interval_float,
        default=0.0,
        help=(
            "on F1B2 cache hits, select this fraction of most-unique even target tokens "
            "for selective refill (mode-dependent); 0 disables (default)"
        ),
    )
    parser.add_argument(
        "--shardedit-selective-refill-mode",
        choices=SELECTIVE_REFILL_MODES,
        default=DEFAULT_SELECTIVE_REFILL_MODE,
        help=(
            "subset=1A gather bridged/run middle_end/scatter; "
            "subset-f1=recompute middle residual from F1 then scatter; "
            "residual-dampen=fixed residual shrink; "
            "uniqueness-scale=shrink residual ∝ uniqueness; "
            "uniqueness-boost=amplify residual ∝ uniqueness"
        ),
    )
    parser.add_argument(
        "--shardedit-selective-refill-dampen",
        type=_unit_interval_float,
        default=DEFAULT_SELECTIVE_REFILL_DAMPEN,
        help=(
            "residual-dampen / uniqueness-scale: shrink amount toward F1; "
            "uniqueness-boost: amplify amount (scale up to 1+dampen); "
            "default 1.0"
        ),
    )
    parser.add_argument(
        "--shardedit-selective-refill-min-step",
        type=_non_negative_int,
        default=0,
        help="only apply selective refill on denoise steps >= this value (0 = all hits)",
    )
    parser.add_argument(
        "--shardedit-cache-threshold",
        type=_unit_interval_float,
        default=0.0,
    )
    parser.add_argument(
        "--shardedit-cache-max-consecutive",
        type=_positive_int,
        default=1,
    )
    parser.add_argument(
        "--shardedit-cache-warmup-steps",
        type=_non_negative_int,
        default=1,
    )
    parser.add_argument(
        "--shardedit-cache-back-blocks",
        type=_non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--shardedit-cache-anchor-mode",
        choices=("residual", "absolute"),
        default=DEFAULT_CACHE_ANCHOR_MODE,
    )
    parser.add_argument(
        "--shardedit-cache-predictor",
        choices=CACHE_PREDICTORS,
        default=DEFAULT_CACHE_PREDICTOR,
    )
    parser.add_argument(
        "--shardedit-cache-threshold-schedule",
        choices=CACHE_THRESHOLD_SCHEDULES,
        default=DEFAULT_CACHE_THRESHOLD_SCHEDULE,
    )
    parser.add_argument(
        "--shardedit-cache-region-policy",
        choices=CACHE_REGION_POLICIES,
        default=DEFAULT_CACHE_REGION_POLICY,
    )
    parser.add_argument(
        "--shardedit-reference-conditioning-size",
        choices=REFERENCE_CONDITIONING_SIZE_POLICIES,
        default=DEFAULT_REFERENCE_CONDITIONING_SIZE,
    )
    parser.add_argument(
        "--shardedit-reference-conditioning-short-side",
        type=_positive_int,
        default=DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE,
    )
    parser.add_argument(
        "--shardedit-reference-conditioning-max-width",
        type=_positive_int,
        default=DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH,
    )
    parser.add_argument(
        "--shardedit-reference-conditioning-max-height",
        type=_positive_int,
        default=DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT,
    )
    parser.add_argument(
        "--shardedit-residency",
        choices=("none", "shard", "window"),
        default=DEFAULT_RESIDENCY_MODE,
    )
    parser.add_argument(
        "--shardedit-residency-window-size",
        type=_positive_int,
        default=DEFAULT_RESIDENCY_WINDOW_SIZE,
    )
    parser.add_argument(
        "--shardedit-release-policy",
        choices=RELEASE_POLICIES,
        default=DEFAULT_RELEASE_POLICY,
        help=(
            "Residency release probe: clear MLX cache after every window, once "
            "after each denoise step, or never explicitly"
        ),
    )
    parser.add_argument(
        "--shardedit-dense-img-ff-window",
        action="store_true",
        help=(
            "After each residency window load+LoRA, dequantize image MLP to dense "
            "bf16 once for that window (opt-in; raises peak memory)"
        ),
    )
    parser.add_argument(
        "--shardedit-dense-img-ff-cache-max-blocks",
        type=_positive_int,
        default=DEFAULT_DENSE_IMG_FF_CACHE_MAX_BLOCKS,
        help=(
            "FIFO cap for cross-step dense img_ff weight reuse "
            "(default covers the full 60-block Transformer)"
        ),
    )
    parser.add_argument(
        "--shardedit-kquant-img-ff-window",
        action="store_true",
        help=(
            "Diagnostic no-go probe: re-encode image MLP weights to mlx-kquant "
            "and reuse them across residency windows/steps"
        ),
    )
    parser.add_argument(
        "--shardedit-kquant-img-ff-cache-max-blocks",
        type=_positive_int,
        default=DEFAULT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS,
        help="FIFO cap for cross-step K-quant img_ff module reuse",
    )
    parser.add_argument(
        "--shardedit-kquant-img-ff-codec",
        default=DEFAULT_KQUANT_IMG_FF_CODEC,
        help="mlx-kquant codec for image MLP weights (default: q6_k)",
    )
    parser.add_argument(
        "--shardedit-lora-tensor-cache",
        action="store_true",
        help=(
            "Opt-in probe: keep per-window LoRA tensors after first load and "
            "reuse them across denoise steps"
        ),
    )
    parser.add_argument(
        "--shardedit-lora-tensor-cache-max-windows",
        type=_positive_int,
        default=DEFAULT_LORA_TENSOR_CACHE_MAX_WINDOWS,
        help="FIFO cap for cached per-window LoRA tensor sets",
    )
    parser.add_argument(
        "--shardedit-patched-window-cache-max-windows",
        type=_non_negative_int,
        default=0,
        help=(
            "Opt-in probe: cache already loaded+LoRA-patched Transformer windows; "
            "0 disables it"
        ),
    )
    parser.add_argument(
        "--shardedit-condition-token-merge",
        action="store_true",
        help=(
            "Diagnostic/rejected V0: locally merge only reference-condition "
            "image tokens while middle full-miss Transformer blocks run, then "
            "unmerge back to the original token count"
        ),
    )
    parser.add_argument(
        "--shardedit-condition-token-merge-stride",
        type=_positive_int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_STRIDE,
        help="Local horizontal condition-token merge stride (default 2)",
    )
    parser.add_argument(
        "--shardedit-condition-token-merge-start-block",
        type=_positive_int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_START_BLOCK,
        help="One-based first Transformer block eligible for condition-token merge",
    )
    parser.add_argument(
        "--shardedit-condition-token-merge-back-blocks",
        type=_non_negative_int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_BACK_BLOCKS,
        help="Keep this many final Transformer blocks unmerged (default 2)",
    )
    parser.add_argument(
        "--shardedit-text-token-merge",
        action="store_true",
        help=(
            "Diagnostic V0: locally merge prompt/VL text tokens "
            "while middle full-miss Transformer blocks run, then unmerge back "
            "to the original token count; smoke showed no speedup"
        ),
    )
    parser.add_argument(
        "--shardedit-text-token-merge-stride",
        type=_positive_int,
        default=DEFAULT_TEXT_TOKEN_MERGE_STRIDE,
        help="Local text-token merge stride (default 2)",
    )
    parser.add_argument(
        "--shardedit-text-token-merge-start-block",
        type=_positive_int,
        default=DEFAULT_TEXT_TOKEN_MERGE_START_BLOCK,
        help="One-based first Transformer block eligible for text-token merge",
    )
    parser.add_argument(
        "--shardedit-text-token-merge-back-blocks",
        type=_non_negative_int,
        default=DEFAULT_TEXT_TOKEN_MERGE_BACK_BLOCKS,
        help="Keep this many final Transformer blocks text-unmerged (default 2)",
    )
    parser.add_argument(
        "--shardedit-q6-linear-profile",
        action="store_true",
        help=(
            "Diagnostic: synchronize and aggregate QuantizedLinear calls inside "
            "executed Qwen Transformer blocks"
        ),
    )
    parser.add_argument("--shardedit-profile", action="store_true")
    namespace, remaining = parser.parse_known_args(argv)
    if namespace.shardedit_cache_threshold > 0.0 and namespace.shardedit_probe_blocks:
        parser.error("residual probing and residual caching cannot be enabled together")
    if namespace.shardedit_bridge_error_diagnose and namespace.shardedit_cache_back_blocks < 1:
        parser.error("--shardedit-bridge-error-diagnose requires --shardedit-cache-back-blocks >= 1")
    if namespace.shardedit_bridge_error_diagnose and namespace.shardedit_cache_threshold > 0.0:
        parser.error(
            "--shardedit-bridge-error-diagnose needs a full pass; set --shardedit-cache-threshold 0"
        )
    if (
        namespace.shardedit_selective_refill_fraction > 0.0
        and namespace.shardedit_cache_threshold <= 0.0
    ):
        parser.error("--shardedit-selective-refill-fraction requires --shardedit-cache-threshold > 0")
    if (
        namespace.shardedit_selective_refill_fraction > 0.0
        and namespace.shardedit_cache_back_blocks < 1
    ):
        parser.error("--shardedit-selective-refill-fraction requires --shardedit-cache-back-blocks >= 1")
    if namespace.shardedit_reference_conditioning_size == "fit-box" and (
        namespace.shardedit_reference_conditioning_max_width < 32
        or namespace.shardedit_reference_conditioning_max_height < 32
    ):
        parser.error("fit-box reference conditioning max dimensions must be >= 32")
    if namespace.shardedit_dense_img_ff_window and namespace.shardedit_residency == "none":
        parser.error("--shardedit-dense-img-ff-window requires residency shard or window")
    if namespace.shardedit_kquant_img_ff_window and namespace.shardedit_residency == "none":
        parser.error("--shardedit-kquant-img-ff-window requires residency shard or window")
    if namespace.shardedit_dense_img_ff_window and namespace.shardedit_kquant_img_ff_window:
        parser.error(
            "--shardedit-dense-img-ff-window and --shardedit-kquant-img-ff-window are mutually exclusive"
        )
    if namespace.shardedit_lora_tensor_cache and namespace.shardedit_residency == "none":
        parser.error("--shardedit-lora-tensor-cache requires residency shard or window")
    if (
        namespace.shardedit_patched_window_cache_max_windows > 0
        and namespace.shardedit_residency == "none"
    ):
        parser.error(
            "--shardedit-patched-window-cache-max-windows requires residency shard or window"
        )
    if namespace.shardedit_condition_token_merge_stride < 2:
        parser.error("--shardedit-condition-token-merge-stride must be >= 2")
    if namespace.shardedit_condition_token_merge_start_block < 1:
        parser.error("--shardedit-condition-token-merge-start-block must be >= 1")
    if namespace.shardedit_text_token_merge_stride < 2:
        parser.error("--shardedit-text-token-merge-stride must be >= 2")
    if namespace.shardedit_text_token_merge_start_block < 1:
        parser.error("--shardedit-text-token-merge-start-block must be >= 1")
    return (
        RuntimeOptions(
            eval_every_n_blocks=namespace.shardedit_eval_every_n_blocks,
            probe_blocks=namespace.shardedit_probe_blocks,
            token_redundancy_blocks=namespace.shardedit_token_redundancy_blocks,
            token_redundancy_heatmap_dir=namespace.shardedit_token_redundancy_heatmap_dir,
            bridge_error_diagnose=namespace.shardedit_bridge_error_diagnose,
            bridge_error_heatmap_dir=namespace.shardedit_bridge_error_heatmap_dir,
            selective_refill_fraction=namespace.shardedit_selective_refill_fraction,
            selective_refill_mode=namespace.shardedit_selective_refill_mode,
            selective_refill_dampen=namespace.shardedit_selective_refill_dampen,
            selective_refill_min_step=namespace.shardedit_selective_refill_min_step,
            cache_threshold=namespace.shardedit_cache_threshold,
            cache_max_consecutive=namespace.shardedit_cache_max_consecutive,
            cache_warmup_steps=namespace.shardedit_cache_warmup_steps,
            cache_back_blocks=namespace.shardedit_cache_back_blocks,
            cache_anchor_mode=namespace.shardedit_cache_anchor_mode,
            cache_predictor=namespace.shardedit_cache_predictor,
            cache_threshold_schedule=namespace.shardedit_cache_threshold_schedule,
            cache_region_policy=namespace.shardedit_cache_region_policy,
            reference_conditioning_size=namespace.shardedit_reference_conditioning_size,
            reference_conditioning_short_side=namespace.shardedit_reference_conditioning_short_side,
            reference_conditioning_max_width=namespace.shardedit_reference_conditioning_max_width,
            reference_conditioning_max_height=namespace.shardedit_reference_conditioning_max_height,
            residency_mode=namespace.shardedit_residency,
            residency_window_size=namespace.shardedit_residency_window_size,
            release_policy=namespace.shardedit_release_policy,
            dense_img_ff_window=namespace.shardedit_dense_img_ff_window,
            dense_img_ff_cache_max_blocks=namespace.shardedit_dense_img_ff_cache_max_blocks,
            kquant_img_ff_window=namespace.shardedit_kquant_img_ff_window,
            kquant_img_ff_cache_max_blocks=namespace.shardedit_kquant_img_ff_cache_max_blocks,
            kquant_img_ff_codec=namespace.shardedit_kquant_img_ff_codec,
            lora_tensor_cache=namespace.shardedit_lora_tensor_cache,
            lora_tensor_cache_max_windows=namespace.shardedit_lora_tensor_cache_max_windows,
            patched_window_cache_max_windows=namespace.shardedit_patched_window_cache_max_windows,
            condition_token_merge=namespace.shardedit_condition_token_merge,
            condition_token_merge_stride=namespace.shardedit_condition_token_merge_stride,
            condition_token_merge_start_block=namespace.shardedit_condition_token_merge_start_block,
            condition_token_merge_back_blocks=namespace.shardedit_condition_token_merge_back_blocks,
            text_token_merge=namespace.shardedit_text_token_merge,
            text_token_merge_stride=namespace.shardedit_text_token_merge_stride,
            text_token_merge_start_block=namespace.shardedit_text_token_merge_start_block,
            text_token_merge_back_blocks=namespace.shardedit_text_token_merge_back_blocks,
            q6_linear_profile=namespace.shardedit_q6_linear_profile,
            profile=namespace.shardedit_profile or namespace.shardedit_q6_linear_profile,
        ),
        remaining,
    )


def should_materialize_block(block_index: int, every_n_blocks: int) -> bool:
    """Return whether this zero-based block closes a materialization group."""

    return every_n_blocks > 0 and (block_index + 1) % every_n_blocks == 0


def should_materialize_residual_anchor(
    *,
    block_index: int,
    cache_enabled: bool,
    cache_hit: bool,
    middle_end_index: int | None,
) -> bool:
    """Return whether this block closes a full-miss residual cache anchor."""

    return (
        cache_enabled
        and not cache_hit
        and middle_end_index is not None
        and block_index in (0, middle_end_index)
    )


def select_middle_anchor_outputs(
    *,
    anchor_mode: str,
    block_index: int,
    encoder_input: Any,
    hidden_input: Any,
    cached_middle_encoder_anchor: Any,
    cached_middle_hidden_anchor: Any,
) -> tuple[Any, Any]:
    """Bridge skipped middle blocks with either a residual delta or absolute state."""

    if block_index != 1:
        return encoder_input, hidden_input
    if anchor_mode == "absolute":
        return cached_middle_encoder_anchor, cached_middle_hidden_anchor
    return (
        encoder_input + cached_middle_encoder_anchor,
        hidden_input + cached_middle_hidden_anchor,
    )


def format_timing_event(name: str, seconds: float, **details: Any) -> str:
    """Format one stable, machine-readable profiling event."""

    event = {"name": name, "seconds": round(seconds, 6), **details}
    return f"SHARDEDIT_TIMING {json.dumps(event, sort_keys=True)}"


def select_guided_noise(
    noise: Any,
    noise_negative: Any,
    guidance: float,
    fallback: GuidanceFunction,
) -> Any:
    """Return positive noise at unit guidance, otherwise use mflux guidance."""

    if guidance == 1.0:
        return noise
    return fallback(noise, noise_negative, guidance)


@contextmanager
def install_runtime_overrides(options: RuntimeOptions = RuntimeOptions()) -> Iterator[None]:
    """Temporarily install qwen-image-shardedit-mlx optimizations in the mflux CLI process."""

    try:
        import mlx.core as mx
        from mlx.nn.layers.quantized import QuantizedLinear
        from mflux.models.common.latent_creator.latent_creator import LatentCreator
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
        from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
        from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        from mflux.utils.generated_image import GeneratedImage
        from mflux.utils.image_util import ImageUtil
        from shardedit_mlx.shard_runtime import ResidencyWindowResult, ShardTransformerRuntime
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise SystemExit("mflux>=0.18 is required to run shardedit-mflux-edit") from exc

    original_guidance = QwenImage.compute_guided_noise
    original_quantized_linear_call = QuantizedLinear.__call__
    original_transformer_call = QwenTransformer.__call__
    original_block = QwenTransformer._apply_transformer_block
    original_model_init = QwenImageEdit.__init__
    original_generate = QwenImageEdit.generate_image
    original_encode_prompts = QwenImageEdit._encode_prompts_with_images
    original_encode_image = LatentCreator.encode_image
    original_create_conditioning = QwenEditUtil.create_image_conditioning_latents
    original_decode = VAEUtil.decode
    original_to_image = ImageUtil.to_image
    original_save = GeneratedImage.save
    call_state = TransformerCallState()
    last_transformer_key: tuple[int, int] | None = None
    last_transformer_output: Any = None
    current_step: int | None = None
    block_zero_inputs: tuple[Any, Any] | None = None
    current_probe_residuals: dict[int, tuple[Any, Any]] = {}
    previous_probe_residuals: dict[int, tuple[Any, Any]] = {}
    residual_cache_state = ResidualCacheState()
    cached_fn_residual: Any = None
    cached_middle_residual: Any = None
    cached_middle_encoder_residual: Any = None
    previous_cached_fn_residual: Any = None
    previous_cached_middle_residual: Any = None
    previous_cached_middle_encoder_residual: Any = None
    older_cached_fn_residual: Any = None
    older_cached_middle_residual: Any = None
    older_cached_middle_encoder_residual: Any = None
    cached_anchor_coordinate: float | None = None
    previous_anchor_coordinate: float | None = None
    older_anchor_coordinate: float | None = None
    prediction_error_ema: float | None = None
    current_fn_hidden: Any = None
    current_fn_encoder: Any = None
    current_fn_residual: Any = None
    current_middle_residual: Any = None
    current_middle_encoder_residual: Any = None
    current_predicted_fn_residual: Any = None
    current_predicted_middle_residual: Any = None
    current_predicted_middle_encoder_residual: Any = None
    current_cache_enabled = False
    current_cache_hit = False
    current_cache_reason = "disabled"
    current_cache_relative_l1: float | None = None
    current_cache_effective_threshold = options.cache_threshold
    current_cache_threshold_progress: float | None = None
    current_cache_threshold_coordinate = "fixed"
    current_cache_veto_threshold: float | None = None
    current_cache_vetoed = False
    current_cache_prediction_scale: float | None = None
    current_cache_prediction: AnchorPrediction | None = None
    current_cache_metrics: dict[str, float | str | None] = {}
    current_cache_coordinate: float | None = None
    current_cache_coordinate_kind = "step"
    current_total_steps = 0
    current_first_sigma: float | None = None
    current_final_sigma: float | None = None
    current_last_block_index: int | None = None
    current_middle_end_index: int | None = None
    current_target_token_count: int | None = None
    current_target_grid_shape: tuple[int, int] | None = None
    current_cond_image_grid: Any = None
    current_text_token_count: int | None = None
    current_valid_text_token_count: int | None = None
    diagnose_previous_middle_residual: Any = None
    diagnose_previous_coordinate: float | None = None
    diagnose_f1_uniqueness: list[float] | None = None
    selective_refill_indices: tuple[int, ...] | None = None
    selective_refill_scales: tuple[float, ...] | None = None
    streaming_runtime: ShardTransformerRuntime | None = None
    prompt_encoding_cache: dict[PromptEncodingCacheKey, tuple[Any, Any, Any, Any]] = {}
    conditioning_latents_cache: dict[
        ConditioningLatentsCacheKey,
        tuple[Any, Any, int, int, int],
    ] = {}
    q6_linear_profiler = Q6LinearProfiler()

    def emit_seconds(name: str, seconds: float, **details: Any) -> None:
        if not options.profile and not options.q6_linear_profile:
            return
        details.setdefault("active_memory_gb", round(mx.get_active_memory() / 1e9, 3))
        details.setdefault("peak_memory_gb", round(mx.get_peak_memory() / 1e9, 3))
        print(format_timing_event(name, seconds, **details), flush=True)

    def emit(name: str, started_at: float, **details: Any) -> None:
        emit_seconds(name, time.perf_counter() - started_at, **details)

    def profiled_quantized_linear_call(layer: Any, x: Any) -> Any:
        if not options.q6_linear_profile or not q6_linear_profiler.should_record(layer):
            return original_quantized_linear_call(layer, x)
        started_at = time.perf_counter()
        output = original_quantized_linear_call(layer, x)
        mx.eval(output)
        q6_linear_profiler.record_call(
            layer,
            input_value=x,
            output_value=output,
            seconds=time.perf_counter() - started_at,
        )
        return output

    def scalar_value(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except (TypeError, ValueError):
            return None

    def rounded_optional(value: Any, digits: int = 8) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return round(parsed, digits)

    def scheduler_sigma(config: Any, index: int) -> float | None:
        sigmas = getattr(getattr(config, "scheduler", None), "sigmas", None)
        if sigmas is None:
            return None
        try:
            return scalar_value(sigmas[index])
        except (IndexError, TypeError):
            return None

    def cache_coordinate(config: Any, step: int) -> tuple[float, str]:
        sigma = scheduler_sigma(config, step - 1)
        if sigma is not None:
            return sigma, "sigma"
        return float(step), "step"

    def token_axis(value: Any) -> int | None:
        shape = getattr(value, "shape", ())
        if len(shape) < 2:
            return None
        return 1 if len(shape) >= 3 else 0

    def token_slice(value: Any, start: int, end: int) -> Any | None:
        axis = token_axis(value)
        if axis is None:
            return None
        shape = getattr(value, "shape", ())
        total_tokens = int(shape[axis])
        start = max(0, min(start, total_tokens))
        end = max(start, min(end, total_tokens))
        if start == end:
            return None
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(start, end)
        return value[tuple(slices)]

    def uniform_valid_text_token_count(mask: Any, total_tokens: int) -> int | None:
        if total_tokens <= 0:
            return None
        if mask is None:
            return total_tokens
        shape = getattr(mask, "shape", ())
        if len(shape) != 2 or int(shape[1]) < total_tokens:
            return None
        lengths = []
        for batch_index in range(int(shape[0])):
            length = int(mx.sum(mask[batch_index, :total_tokens]).item())
            lengths.append(length)
        if not lengths or any(length != lengths[0] for length in lengths):
            return None
        valid_tokens = lengths[0]
        if valid_tokens <= 0 or valid_tokens > total_tokens:
            return None
        return valid_tokens

    def bipartite_best_match_similarities(region: Any) -> list[float] | None:
        """Cosine-similarity proxy for how mergeable a token region is (ToMe-style).

        Splits tokens into even ("src") / odd ("dst") halves along the token
        axis -- a simplification of ToMe's checkerboard partition, adequate
        for a diagnostic -- and returns each src token's best cosine
        similarity to any dst token. Diagnostic only: does not merge, skip,
        or otherwise change any block computation.
        """

        axis = token_axis(region)
        if axis is None:
            return None
        shape = getattr(region, "shape", ())
        total_tokens = int(shape[axis])
        if total_tokens < 4:
            return None
        src_slices = [slice(None)] * len(shape)
        dst_slices = [slice(None)] * len(shape)
        src_slices[axis] = slice(0, None, 2)
        dst_slices[axis] = slice(1, None, 2)
        src = region[tuple(src_slices)]
        dst = region[tuple(dst_slices)]
        src_norm = src / (mx.sqrt(mx.sum(src * src, axis=-1, keepdims=True)) + 1e-8)
        dst_norm = dst / (mx.sqrt(mx.sum(dst * dst, axis=-1, keepdims=True)) + 1e-8)
        if axis == 1:
            similarity = src_norm[0] @ dst_norm[0].T
        else:
            similarity = src_norm @ dst_norm.T
        best = mx.max(similarity, axis=-1)
        mx.eval(best)
        return [float(value) for value in best.tolist()]

    def save_token_redundancy_heatmap(
        *,
        best: list[float],
        grid_shape: tuple[int, int],
        region_name: str,
        block: int,
    ) -> None:
        heatmap_dir = options.token_redundancy_heatmap_dir
        if heatmap_dir is None:
            return
        try:
            grid = similarities_to_grid(
                best,
                grid_height=grid_shape[0],
                grid_width=grid_shape[1],
            )
        except ValueError:
            return
        from PIL import Image

        rows = [[similarity_to_rgb(value) for value in row] for row in grid]
        height = len(rows)
        width = len(rows[0]) if height else 0
        if height == 0 or width == 0:
            return
        image = Image.new("RGB", (width, height))
        image.putdata([pixel for row in rows for pixel in row])
        # rows only cover the "src" (even-column) half of the real patch grid
        # (see shardedit_mlx.token_redundancy_heatmap docstring) -- restore the
        # real aspect ratio with nearest-neighbor upscaling for visual
        # comparison against the generated image.
        image = image.resize((grid_shape[1], grid_shape[0]), resample=Image.NEAREST)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        out_path = heatmap_dir / f"step{current_step}_block{block}_{region_name}.png"
        image.save(out_path)

    def emit_token_redundancy(
        *,
        region_name: str,
        region: Any,
        block: int,
        started_at: float,
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        best = bipartite_best_match_similarities(region)
        if best is None:
            return
        summary = redundancy_summary(best)
        details = {
            f"fraction_ge_{threshold}": round(fraction, 6)
            for threshold, fraction in summary.fraction_above_threshold.items()
        }
        emit(
            "token_redundancy",
            started_at,
            step=current_step,
            block=block,
            region=region_name,
            token_count=summary.token_count,
            mean_best_similarity=round(summary.mean_best_similarity, 6),
            median_best_similarity=round(summary.median_best_similarity, 6),
            **details,
        )
        if grid_shape is not None:
            save_token_redundancy_heatmap(
                best=best,
                grid_shape=grid_shape,
                region_name=region_name,
                block=block,
            )

    def per_token_mean_abs(predicted: Any, actual: Any) -> list[float] | None:
        axis = token_axis(predicted)
        if axis is None or getattr(predicted, "shape", None) != getattr(actual, "shape", None):
            return None
        abs_err = mx.mean(mx.abs(predicted - actual), axis=-1)
        mx.eval(abs_err)
        flat = abs_err[0] if axis == 1 else abs_err
        return [float(value) for value in flat.tolist()]

    def gather_tokens(value: Any, indices: Sequence[int]) -> Any:
        axis = token_axis(value)
        if axis is None:
            raise RuntimeError("cannot gather tokens without a token axis")
        index_array = mx.array(list(indices), dtype=mx.int32)
        return mx.take(value, index_array, axis=axis)

    def scatter_tokens(full: Any, gathered: Any, indices: Sequence[int]) -> Any:
        """Write gathered token rows back into a copy of `full` at `indices`."""

        axis = token_axis(full)
        if axis is None:
            raise RuntimeError("cannot scatter tokens without a token axis")
        # Materialize to python lists for a reliable sparse write; token counts
        # are O(10^3) so this is fine for the opt-in diagnostic/refill path.
        full_list = full.tolist()
        gathered_list = gathered.tolist()
        if axis == 1:
            for out_i, token_i in enumerate(indices):
                full_list[0][token_i] = gathered_list[0][out_i]
        else:
            for out_i, token_i in enumerate(indices):
                full_list[token_i] = gathered_list[out_i]
        return mx.array(full_list, dtype=full.dtype)

    def gather_image_rotary(
        image_rotary_embeddings: Any,
        indices: Sequence[int],
    ) -> Any:
        (img_cos, img_sin), txt_pair = image_rotary_embeddings
        index_array = mx.array(list(indices), dtype=mx.int32)
        return (
            (mx.take(img_cos, index_array, axis=0), mx.take(img_sin, index_array, axis=0)),
            txt_pair,
        )

    def merge_condition_hidden_tokens(value: Any, plan: ConditionMergePlan) -> Any:
        axis = token_axis(value)
        if axis != 1:
            raise RuntimeError("condition token merge expects batched image tokens")
        target = token_slice(value, 0, plan.target_token_count)
        condition = token_slice(value, plan.target_token_count, plan.image_token_count)
        if target is None or condition is None:
            raise RuntimeError("condition token merge could not split target/condition tokens")
        merged_parts: list[Any] = []
        offset = 0
        for grid in plan.grids:
            count = grid.token_count
            region = token_slice(condition, offset, offset + count)
            if region is None:
                raise RuntimeError("condition token merge grid slice is empty")
            merged_parts.append(merge_condition_grid_region(region, grid, plan.stride))
            offset += count
        merged_condition = (
            merged_parts[0]
            if len(merged_parts) == 1
            else mx.concatenate(merged_parts, axis=axis)
        )
        return mx.concatenate([target, merged_condition], axis=axis)

    def merge_condition_grid_region(region: Any, grid: ConditionGrid, stride: int) -> Any:
        batch, _, dim = region.shape
        reshaped = mx.reshape(
            region,
            (batch, grid.frames, grid.height, grid.width, dim),
        )
        parts: list[Any] = []
        full_width = (grid.width // stride) * stride
        if full_width:
            head = reshaped[:, :, :, :full_width, :]
            head = mx.reshape(
                head,
                (batch, grid.frames, grid.height, full_width // stride, stride, dim),
            )
            parts.append(mx.mean(head, axis=4))
        tail_width = grid.width - full_width
        if tail_width:
            tail = reshaped[:, :, :, full_width:, :]
            parts.append(mx.mean(tail, axis=3, keepdims=True))
        merged = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=3)
        return mx.reshape(
            merged,
            (batch, grid.frames * grid.height * grid.merged_width(stride=stride), dim),
        )

    def unmerge_condition_hidden_tokens(
        *,
        original_input: Any,
        merged_input: Any,
        merged_output: Any,
        plan: ConditionMergePlan,
    ) -> Any:
        axis = token_axis(original_input)
        if axis != 1:
            raise RuntimeError("condition token unmerge expects batched image tokens")
        target_output = token_slice(merged_output, 0, plan.target_token_count)
        original_condition = token_slice(
            original_input,
            plan.target_token_count,
            plan.image_token_count,
        )
        merged_input_condition = token_slice(
            merged_input,
            plan.target_token_count,
            plan.merged_image_token_count,
        )
        merged_output_condition = token_slice(
            merged_output,
            plan.target_token_count,
            plan.merged_image_token_count,
        )
        if (
            target_output is None
            or original_condition is None
            or merged_input_condition is None
            or merged_output_condition is None
        ):
            raise RuntimeError("condition token unmerge could not split token regions")
        merged_delta = merged_output_condition - merged_input_condition
        full_delta_parts: list[Any] = []
        offset = 0
        merged_offset = 0
        for grid in plan.grids:
            merged_count = grid.merged_token_count(stride=plan.stride)
            delta_region = token_slice(merged_delta, merged_offset, merged_offset + merged_count)
            if delta_region is None:
                raise RuntimeError("condition token unmerge grid delta is empty")
            full_delta_parts.append(unmerge_condition_grid_delta(delta_region, grid, plan.stride))
            offset += grid.token_count
            merged_offset += merged_count
        full_delta = (
            full_delta_parts[0]
            if len(full_delta_parts) == 1
            else mx.concatenate(full_delta_parts, axis=axis)
        )
        if offset != plan.condition_token_count:
            raise RuntimeError("condition token unmerge did not consume every condition token")
        condition_output = original_condition + full_delta
        return mx.concatenate([target_output, condition_output], axis=axis)

    def unmerge_condition_grid_delta(delta: Any, grid: ConditionGrid, stride: int) -> Any:
        batch, _, dim = delta.shape
        merged_width = grid.merged_width(stride=stride)
        reshaped = mx.reshape(delta, (batch, grid.frames, grid.height, merged_width, dim))
        parts: list[Any] = []
        full_groups = grid.width // stride
        if full_groups:
            parts.append(mx.repeat(reshaped[:, :, :, :full_groups, :], repeats=stride, axis=3))
        tail_width = grid.width - full_groups * stride
        if tail_width:
            parts.append(
                mx.repeat(
                    reshaped[:, :, :, full_groups : full_groups + 1, :],
                    repeats=tail_width,
                    axis=3,
                )
            )
        full = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=3)
        return mx.reshape(full, (batch, grid.token_count, dim))

    def merged_condition_rotary_embeddings(
        image_rotary_embeddings: Any,
        plan: ConditionMergePlan,
    ) -> Any:
        return gather_image_rotary(image_rotary_embeddings, merged_image_representative_indices(plan))

    def merged_image_representative_indices(plan: ConditionMergePlan) -> tuple[int, ...]:
        indices = list(range(plan.target_token_count))
        condition_offset = 0
        for grid in plan.grids:
            grid_base = plan.target_token_count + condition_offset
            for frame in range(grid.frames):
                for row in range(grid.height):
                    row_base = grid_base + (frame * grid.height + row) * grid.width
                    for col in range(0, grid.width, plan.stride):
                        indices.append(row_base + col)
            condition_offset += grid.token_count
        return tuple(indices)

    def merge_text_tokens(value: Any, plan: TextMergePlan) -> Any:
        axis = token_axis(value)
        if axis != 1:
            raise RuntimeError("text token merge expects batched text tokens")
        valid = token_slice(value, 0, plan.valid_text_token_count)
        if valid is None:
            raise RuntimeError("text token merge could not slice valid text tokens")
        return merge_text_token_region(valid, plan.stride)

    def merge_text_token_region(region: Any, stride: int) -> Any:
        batch, token_count, dim = region.shape
        parts: list[Any] = []
        full_width = (token_count // stride) * stride
        if full_width:
            head = region[:, :full_width, :]
            head = mx.reshape(head, (batch, full_width // stride, stride, dim))
            parts.append(mx.mean(head, axis=2))
        tail_width = token_count - full_width
        if tail_width:
            tail = region[:, full_width:, :]
            parts.append(mx.mean(tail, axis=1, keepdims=True))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

    def merge_text_mask(mask: Any, plan: TextMergePlan) -> Any:
        if mask is None:
            return None
        shape = getattr(mask, "shape", ())
        if len(shape) != 2:
            raise RuntimeError("text token merge expects a 2D text mask")
        return mx.ones(
            (int(shape[0]), plan.merged_text_token_count),
            dtype=mask.dtype,
        )

    def unmerge_text_tokens(
        *,
        original_input: Any,
        merged_input: Any,
        merged_output: Any,
        plan: TextMergePlan,
    ) -> Any:
        axis = token_axis(original_input)
        if axis != 1:
            raise RuntimeError("text token unmerge expects batched text tokens")
        original_valid = token_slice(original_input, 0, plan.valid_text_token_count)
        if original_valid is None:
            raise RuntimeError("text token unmerge could not slice valid text tokens")
        if int(getattr(merged_input, "shape", (0, 0))[axis]) != plan.merged_text_token_count:
            raise RuntimeError("text token unmerge received an unexpected merged input shape")
        if int(getattr(merged_output, "shape", (0, 0))[axis]) != plan.merged_text_token_count:
            raise RuntimeError("text token unmerge received an unexpected merged output shape")
        merged_delta = merged_output - merged_input
        full_delta = unmerge_text_token_delta(
            merged_delta,
            valid_token_count=plan.valid_text_token_count,
            stride=plan.stride,
        )
        valid_output = original_valid + full_delta
        padding = token_slice(original_input, plan.valid_text_token_count, plan.text_token_count)
        if padding is None:
            return valid_output
        return mx.concatenate([valid_output, padding], axis=axis)

    def unmerge_text_token_delta(delta: Any, *, valid_token_count: int, stride: int) -> Any:
        batch, _, dim = delta.shape
        full_groups = valid_token_count // stride
        parts: list[Any] = []
        if full_groups:
            parts.append(mx.repeat(delta[:, :full_groups, :], repeats=stride, axis=1))
        tail_width = valid_token_count - full_groups * stride
        if tail_width:
            parts.append(
                mx.repeat(
                    delta[:, full_groups : full_groups + 1, :],
                    repeats=tail_width,
                    axis=1,
                )
            )
        full = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        return mx.reshape(full, (batch, valid_token_count, dim))

    def merged_text_rotary_embeddings(
        image_rotary_embeddings: Any,
        plan: TextMergePlan,
    ) -> Any:
        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_embeddings
        index_array = mx.array(text_representative_indices(plan), dtype=mx.int32)
        return (
            (img_cos, img_sin),
            (mx.take(txt_cos, index_array, axis=0), mx.take(txt_sin, index_array, axis=0)),
        )

    def text_representative_indices(plan: TextMergePlan) -> tuple[int, ...]:
        return tuple(range(0, plan.valid_text_token_count, plan.stride))

    def save_bridge_error_heatmap(abs_errors_even: Sequence[float], *, block: int) -> None:
        heatmap_dir = options.bridge_error_heatmap_dir
        if heatmap_dir is None or current_target_grid_shape is None:
            return
        # Map abs error to a 0..1 score via a robust percentile stretch, then reuse
        # the red=high / blue=low diverging palette by inverting (high error = red).
        values = list(abs_errors_even)
        if not values:
            return
        lo = min(values)
        hi = max(values)
        span = hi - lo if hi > lo else 1.0
        uniqueness_like = [(hi - value) / span for value in values]
        try:
            grid = similarities_to_grid(
                uniqueness_like,
                grid_height=current_target_grid_shape[0],
                grid_width=current_target_grid_shape[1],
            )
        except ValueError:
            return
        from PIL import Image

        rows = [[similarity_to_rgb(value) for value in row] for row in grid]
        image = Image.new("RGB", (len(rows[0]), len(rows)))
        image.putdata([pixel for row in rows for pixel in row])
        image = image.resize(
            (current_target_grid_shape[1], current_target_grid_shape[0]),
            resample=Image.NEAREST,
        )
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        image.save(heatmap_dir / f"step{current_step}_block{block}_bridge_error.png")

    def emit_bridge_error_diagnosis(
        *,
        predicted_middle: Any,
        actual_middle: Any,
        uniqueness: Sequence[float],
        block: int,
        started_at: float,
    ) -> None:
        target_count = current_target_token_count
        if target_count is None or target_count <= 0:
            return
        predicted_target = token_slice(predicted_middle, 0, target_count)
        actual_target = token_slice(actual_middle, 0, target_count)
        if predicted_target is None or actual_target is None:
            return
        abs_errors = per_token_mean_abs(predicted_target, actual_target)
        if abs_errors is None:
            return
        even_errors = even_index_values(abs_errors)
        if len(even_errors) != len(uniqueness):
            return
        uniq = uniqueness_from_similarities(uniqueness)
        summary = correlate_bridge_error_with_uniqueness(
            abs_errors=even_errors,
            uniqueness=uniq,
        )
        emit(
            "bridge_error_vs_redundancy",
            started_at,
            step=current_step,
            block=block,
            token_count=summary.token_count,
            mean_abs_error=round(summary.mean_abs_error, 6),
            mean_uniqueness=round(summary.mean_uniqueness, 6),
            pearson=round(summary.pearson, 6),
            spearman=round(summary.spearman, 6),
            go=summary.go,
        )
        save_bridge_error_heatmap(even_errors, block=block)

    def scale_bridged_residuals(
        bridged: Any,
        f1: Any,
        indices: Sequence[int],
        scales: float | Sequence[float],
    ) -> Any:
        """Apply ``f1 + scale * (bridged - f1)`` on selected tokens."""

        bridged_g = gather_tokens(bridged, indices)
        f1_g = gather_tokens(f1, indices)
        residual_g = bridged_g - f1_g
        if isinstance(scales, (int, float)):
            scaled = f1_g + float(scales) * residual_g
        else:
            if len(scales) != len(indices):
                raise RuntimeError("per-token scales must match selected indices")
            axis = token_axis(bridged_g)
            if axis is None:
                raise RuntimeError("cannot apply per-token scale without a token axis")
            scale_arr = mx.array(list(scales), dtype=bridged_g.dtype)
            shape = [1] * len(getattr(bridged_g, "shape", ()))
            shape[axis] = len(scales)
            scale_arr = mx.reshape(scale_arr, shape)
            scaled = f1_g + scale_arr * residual_g
        return scatter_tokens(bridged, scaled, indices)

    def maybe_apply_residual_dampen(
        *,
        block_index: int,
        encoder_bridged: Any,
        hidden_bridged: Any,
        hidden_f1: Any,
    ) -> tuple[Any, Any] | None:
        """On cache hit at block 1, scale middle residual on unique target tokens.

        Only the image hidden stream is adjusted; the text encoder stream keeps the
        full residual bridge (same as subset 1A).
        """

        if (
            options.selective_refill_mode not in RESIDUAL_ADJUST_MODES
            or block_index != 1
            or selective_refill_indices is None
            or not should_apply_selective_refill(
                fraction=options.selective_refill_fraction,
                current_step=current_step,
                min_step=options.selective_refill_min_step,
                cache_hit=current_cache_hit,
            )
        ):
            return None
        started_at = time.perf_counter()
        if options.selective_refill_mode in ("uniqueness-scale", "uniqueness-boost"):
            if selective_refill_scales is None:
                return None
            scales: float | Sequence[float] = selective_refill_scales
            mean_scale = sum(selective_refill_scales) / max(len(selective_refill_scales), 1)
            max_scale = max(selective_refill_scales) if selective_refill_scales else 1.0
            min_scale = min(selective_refill_scales) if selective_refill_scales else 1.0
        else:
            # residual-dampen: scale = 1 - dampen
            scales = 1.0 - options.selective_refill_dampen
            mean_scale = float(scales)
            max_scale = float(scales)
            min_scale = float(scales)
        refined_hidden = scale_bridged_residuals(
            hidden_bridged, hidden_f1, selective_refill_indices, scales
        )
        emit(
            "selective_refill",
            started_at,
            step=current_step,
            block=block_index + 1,
            mode=options.selective_refill_mode,
            selected_target_tokens=len(selective_refill_indices),
            fraction=options.selective_refill_fraction,
            dampen=options.selective_refill_dampen,
            mean_scale=round(mean_scale, 6),
            max_scale=round(max_scale, 6),
            min_scale=round(min_scale, 6),
            min_step=options.selective_refill_min_step,
        )
        return encoder_bridged, refined_hidden

    def maybe_run_selective_middle_refill(
        *,
        block_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        encoder_input: Any,
        hidden_input: Any,
    ) -> tuple[Any, Any] | None:
        """On cache hit at middle_end, refine unique tokens through one real block.

        ``subset`` gathers the current (bridged) hidden state. ``subset-f1`` gathers
        the F1 hidden state so the block recomputes a middle residual from the
        pre-bridge anchor, then scatters onto the bridged full sequence.
        """

        if (
            options.selective_refill_mode not in SUBSET_REFILL_MODES
            or current_middle_end_index is None
            or block_index != current_middle_end_index
            or selective_refill_indices is None
            or current_target_token_count is None
            or not should_apply_selective_refill(
                fraction=options.selective_refill_fraction,
                current_step=current_step,
                min_step=options.selective_refill_min_step,
                cache_hit=current_cache_hit,
            )
        ):
            return None
        source_hidden = hidden_input
        if options.selective_refill_mode == "subset-f1":
            if current_fn_hidden is None:
                return None
            source_hidden = current_fn_hidden
        axis = token_axis(hidden_input)
        if axis is None:
            return None
        total_tokens = int(getattr(hidden_input, "shape", ())[axis])
        gather_indices = build_image_gather_indices(
            selective_refill_indices,
            target_token_count=current_target_token_count,
            total_image_tokens=total_tokens,
        )
        image_rotary = kwargs.get(
            "image_rotary_embeddings",
            args[6] if len(args) > 6 else None,
        )
        if image_rotary is None:
            return None
        started_at = time.perf_counter()
        gathered_hidden = gather_tokens(source_hidden, gather_indices)
        gathered_rotary = gather_image_rotary(image_rotary, gather_indices)
        call_kwargs = dict(kwargs)
        call_kwargs["hidden_states"] = gathered_hidden
        call_kwargs["encoder_hidden_states"] = encoder_input
        call_kwargs["image_rotary_embeddings"] = gathered_rotary
        call_args = list(args)
        if len(call_args) > 2:
            call_args[2] = gathered_hidden
        if len(call_args) > 3:
            call_args[3] = encoder_input
        if len(call_args) > 6:
            call_args[6] = gathered_rotary
        encoder_out, hidden_out = original_block(*call_args, **call_kwargs)
        refined_hidden = scatter_tokens(hidden_input, hidden_out, gather_indices)
        emit(
            "selective_refill",
            started_at,
            step=current_step,
            block=block_index + 1,
            mode=options.selective_refill_mode,
            selected_target_tokens=len(selective_refill_indices),
            gathered_image_tokens=len(gather_indices),
            fraction=options.selective_refill_fraction,
            min_step=options.selective_refill_min_step,
        )
        return encoder_input, refined_hidden

    def maybe_run_token_merged_block(
        *,
        block_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        encoder_input: Any,
        hidden_input: Any,
    ) -> tuple[Any, Any] | None:
        """Run one full-miss middle block with optional shortened token streams."""

        if current_target_token_count is None or current_last_block_index is None:
            return None
        image_rotary = kwargs.get(
            "image_rotary_embeddings",
            args[6] if len(args) > 6 else None,
        )
        if image_rotary is None:
            return None

        condition_plan = None
        if should_merge_condition_block(
            enabled=options.condition_token_merge,
            cache_hit=current_cache_hit,
            block_index=block_index,
            block_count=current_last_block_index + 1,
            start_block=options.condition_token_merge_start_block,
            back_blocks=options.condition_token_merge_back_blocks,
        ):
            axis = token_axis(hidden_input)
            if axis == 1:
                total_tokens = int(getattr(hidden_input, "shape", ())[axis])
                condition_plan = build_condition_merge_plan(
                    target_token_count=current_target_token_count,
                    total_image_tokens=total_tokens,
                    cond_image_grid=current_cond_image_grid,
                    stride=options.condition_token_merge_stride,
                )
                if (
                    condition_plan is not None
                    and condition_plan.merged_condition_token_count
                    >= condition_plan.condition_token_count
                ):
                    condition_plan = None

        text_plan = None
        if should_merge_condition_block(
            enabled=options.text_token_merge,
            cache_hit=current_cache_hit,
            block_index=block_index,
            block_count=current_last_block_index + 1,
            start_block=options.text_token_merge_start_block,
            back_blocks=options.text_token_merge_back_blocks,
        ):
            if current_text_token_count is not None and current_valid_text_token_count is not None:
                text_plan = build_text_merge_plan(
                    total_text_tokens=current_text_token_count,
                    valid_text_tokens=current_valid_text_token_count,
                    stride=options.text_token_merge_stride,
                )

        if condition_plan is None and text_plan is None:
            return None

        started_at = time.perf_counter()
        merged_hidden = hidden_input
        merged_encoder = encoder_input
        merged_mask = kwargs.get(
            "encoder_hidden_states_mask",
            args[4] if len(args) > 4 else None,
        )
        merged_rotary = image_rotary
        if condition_plan is not None:
            merged_hidden = merge_condition_hidden_tokens(hidden_input, condition_plan)
            merged_rotary = merged_condition_rotary_embeddings(merged_rotary, condition_plan)
        if text_plan is not None:
            merged_encoder = merge_text_tokens(encoder_input, text_plan)
            merged_mask = merge_text_mask(merged_mask, text_plan)
            merged_rotary = merged_text_rotary_embeddings(merged_rotary, text_plan)
        call_kwargs = dict(kwargs)
        call_kwargs["hidden_states"] = merged_hidden
        call_kwargs["encoder_hidden_states"] = merged_encoder
        call_kwargs["encoder_hidden_states_mask"] = merged_mask
        call_kwargs["image_rotary_embeddings"] = merged_rotary
        call_args = list(args)
        if len(call_args) > 2:
            call_args[2] = merged_hidden
        if len(call_args) > 3:
            call_args[3] = merged_encoder
        if len(call_args) > 4:
            call_args[4] = merged_mask
        if len(call_args) > 6:
            call_args[6] = merged_rotary
        merged_encoder_output, merged_hidden_output = original_block(*call_args, **call_kwargs)
        encoder_output = merged_encoder_output
        hidden_output = merged_hidden_output
        if condition_plan is not None:
            hidden_output = unmerge_condition_hidden_tokens(
                original_input=hidden_input,
                merged_input=merged_hidden,
                merged_output=merged_hidden_output,
                plan=condition_plan,
            )
            emit(
                "condition_token_merge",
                started_at,
                step=current_step,
                block=block_index + 1,
                stride=condition_plan.stride,
                image_tokens_before=condition_plan.image_token_count,
                image_tokens_after=condition_plan.merged_image_token_count,
                condition_tokens_before=condition_plan.condition_token_count,
                condition_tokens_after=condition_plan.merged_condition_token_count,
                reduction_ratio=round(condition_plan.reduction_ratio, 6),
            )
        if text_plan is not None:
            encoder_output = unmerge_text_tokens(
                original_input=encoder_input,
                merged_input=merged_encoder,
                merged_output=merged_encoder_output,
                plan=text_plan,
            )
            emit(
                "text_token_merge",
                started_at,
                step=current_step,
                block=block_index + 1,
                stride=text_plan.stride,
                text_tokens_before=text_plan.text_token_count,
                valid_text_tokens_before=text_plan.valid_text_token_count,
                text_tokens_after=text_plan.merged_text_token_count,
                reduction_ratio=round(text_plan.reduction_ratio, 6),
            )
        return encoder_output, hidden_output

    def tensor_relative_l1(predicted: Any, current: Any) -> tuple[Any, Any, Any]:
        abs_l1 = mx.mean(mx.abs(predicted - current))
        denom_l1 = mx.mean(mx.abs(predicted)) + 1e-8
        return abs_l1 / denom_l1, abs_l1, denom_l1

    def maybe_region_relative_l1(
        predicted: Any,
        current: Any,
        *,
        start: int,
        end: int,
    ) -> Any | None:
        predicted_region = token_slice(predicted, start, end)
        current_region = token_slice(current, start, end)
        if predicted_region is None or current_region is None:
            return None
        relative_l1, _, _ = tensor_relative_l1(predicted_region, current_region)
        return relative_l1

    def prediction_metrics(
        predicted: Any,
        current: Any,
        *,
        target_token_count: int | None,
    ) -> dict[str, float | None]:
        relative_l1, abs_l1, denom_l1 = tensor_relative_l1(predicted, current)
        values = [relative_l1, abs_l1, denom_l1]
        target_relative_l1 = None
        condition_relative_l1 = None
        axis = token_axis(predicted)
        if axis is not None and target_token_count is not None and target_token_count > 0:
            total_tokens = int(getattr(predicted, "shape", ())[axis])
            target_relative_l1 = maybe_region_relative_l1(
                predicted,
                current,
                start=0,
                end=target_token_count,
            )
            if target_relative_l1 is not None:
                values.append(target_relative_l1)
            condition_relative_l1 = maybe_region_relative_l1(
                predicted,
                current,
                start=target_token_count,
                end=total_tokens,
            )
            if condition_relative_l1 is not None:
                values.append(condition_relative_l1)

        dot_product = mx.sum(predicted * current)
        predicted_norm = mx.sqrt(mx.sum(predicted * predicted))
        current_norm = mx.sqrt(mx.sum(current * current))
        prediction_cosine = dot_product / (predicted_norm * current_norm + 1e-8)
        magnitude_ratio = mx.mean(mx.abs(current)) / (mx.mean(mx.abs(predicted)) + 1e-8)
        values.extend([prediction_cosine, magnitude_ratio])
        mx.eval(*values)
        return {
            "relative_l1": float(relative_l1.item()),
            "prediction_abs_l1": float(abs_l1.item()),
            "prediction_denom_l1": float(denom_l1.item()),
            "target_relative_l1": (
                float(target_relative_l1.item()) if target_relative_l1 is not None else None
            ),
            "condition_relative_l1": (
                float(condition_relative_l1.item()) if condition_relative_l1 is not None else None
            ),
            "prediction_cosine": float(prediction_cosine.item()),
            "prediction_magnitude_ratio": float(magnitude_ratio.item()),
        }

    def optimized(noise: Any, noise_negative: Any, guidance: float) -> Any:
        return select_guided_noise(noise, noise_negative, guidance, original_guidance)

    def profiled_model_init(model: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal streaming_runtime
        started_at = time.perf_counter()
        original_model_init(model, *args, **kwargs)
        if options.residency_mode != "none":
            residency_started_at = time.perf_counter()
            model_path = kwargs.get("model_path", args[1] if len(args) > 1 else None)
            if model_path is None:
                raise RuntimeError("weight residency requires a local --model path")
            if model.bits != 6:
                raise RuntimeError(
                    f"weight residency currently requires q6 weights, found bits={model.bits}"
                )
            lora_paths = tuple(str(path) for path in (model.lora_paths or ()))
            lora_scales = tuple(float(scale) for scale in (model.lora_scales or ()))
            streaming_runtime = ShardTransformerRuntime.create(
                model_path=Path(model_path).expanduser().resolve(),
                mode=options.residency_mode,
                window_size=options.residency_window_size,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                dense_img_ff=options.dense_img_ff_window,
                dense_img_ff_cache_max_blocks=options.dense_img_ff_cache_max_blocks,
                kquant_img_ff=options.kquant_img_ff_window,
                kquant_img_ff_cache_max_blocks=options.kquant_img_ff_cache_max_blocks,
                kquant_img_ff_codec=options.kquant_img_ff_codec,
                release_policy=options.release_policy,
                lora_tensor_cache=options.lora_tensor_cache,
                lora_tensor_cache_max_windows=options.lora_tensor_cache_max_windows,
                patched_window_cache_max_windows=options.patched_window_cache_max_windows,
            )
            released_blocks = streaming_runtime.detach_resident_blocks(model.transformer)
            emit(
                "residency_init",
                residency_started_at,
                mode=options.residency_mode,
                window_size=(
                    options.residency_window_size
                    if options.residency_mode == "window"
                    else None
                ),
                windows=len(streaming_runtime.windows),
                released_blocks=released_blocks,
                lora_files=len(streaming_runtime.lora_sources),
                lora_keys=streaming_runtime.lora_key_count,
                dense_img_ff_window=options.dense_img_ff_window,
                dense_img_ff_cache_max_blocks=options.dense_img_ff_cache_max_blocks,
                kquant_img_ff_window=options.kquant_img_ff_window,
                kquant_img_ff_cache_max_blocks=options.kquant_img_ff_cache_max_blocks,
                kquant_img_ff_codec=options.kquant_img_ff_codec,
                release_policy=options.release_policy,
                lora_tensor_cache=options.lora_tensor_cache,
                lora_tensor_cache_max_windows=options.lora_tensor_cache_max_windows,
                patched_window_cache_max_windows=options.patched_window_cache_max_windows,
            )
        emit("model_init", started_at)

    def profiled_generate(model: Any, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_generate(model, *args, **kwargs)
        emit("generate_total", started_at)
        return result

    def optimized_encode_prompts(
        model: Any,
        prompt: str,
        negative_prompt: str | None,
        image_paths: list[str],
        config: Any,
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        tokenizer = model.tokenizers.get("qwen_vl")
        use_picture_prefix = infer_prompt_cache_picture_prefix(
            image_paths,
            getattr(tokenizer, "use_picture_prefix", None),
        )

        cache_key = prompt_encoding_cache_key(
            prompt=prompt,
            negative_prompt=negative_prompt if config.guidance != 1.0 else None,
            image_paths=image_paths,
            vl_width=vl_width,
            vl_height=vl_height,
            guidance=float(config.guidance),
            use_picture_prefix=use_picture_prefix,
        )
        cache_started_at = time.perf_counter()
        cached = prompt_encoding_cache.get(cache_key)
        if cached is not None:
            emit(
                "vision_language_cache_hit",
                cache_started_at,
                prompt_tokens=cached[0].shape[1],
                reference_count=len(image_paths),
            )
            return cached

        if tokenizer is None or getattr(model, "qwen_vl_encoder", None) is None:
            raise RuntimeError(
                "Qwen VLM tokenizer/encoder was released and this request was not in "
                "the warm prompt cache"
            )

        if len(image_paths) > 1 and not tokenizer.use_picture_prefix:
            tokenizer = QwenVisionLanguageTokenizer(
                processor=tokenizer.processor,
                max_length=tokenizer.max_length,
                use_picture_prefix=True,
            )
            model.tokenizers["qwen_vl"] = tokenizer

        if config.guidance != 1.0:
            started_at = time.perf_counter()
            outputs = original_encode_prompts(
                model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_paths=image_paths,
                config=config,
                vl_width=vl_width,
                vl_height=vl_height,
            )
            mx.eval(*outputs)
            emit("vision_language_encode_both", started_at, reference_count=len(image_paths))
            prompt_encoding_cache[cache_key] = outputs
            return outputs

        tokenize_started_at = time.perf_counter()
        input_ids, attention_mask, pixel_values, image_grid_thw = tokenizer.tokenize_with_image(
            prompt,
            image_paths,
            vl_width=vl_width,
            vl_height=vl_height,
        )
        emit("vision_language_tokenize", tokenize_started_at, reference_count=len(image_paths))

        encode_started_at = time.perf_counter()
        hidden_states = model.qwen_vl_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        prompt_embeds = hidden_states[0].astype(mx.float16)
        prompt_mask = hidden_states[1].astype(mx.float16)
        mx.eval(prompt_embeds, prompt_mask)
        emit(
            "vision_language_encode_positive",
            encode_started_at,
            prompt_tokens=prompt_embeds.shape[1],
            reference_count=len(image_paths),
        )

        # The negative values are never consumed by the unit-guidance Transformer path.
        outputs = prompt_embeds, prompt_mask, prompt_embeds, prompt_mask
        prompt_encoding_cache[cache_key] = outputs
        return outputs

    def profiled_encode_image(*args: Any, **kwargs: Any) -> Any:
        condition_height, condition_width = vae_encode_condition_dimensions(args, kwargs)
        started_at = time.perf_counter()
        result = original_encode_image(*args, **kwargs)
        mx.eval(result)
        details: dict[str, Any] = {
            "reference_conditioning_size": options.reference_conditioning_size,
            "reference_conditioning_short_side": options.reference_conditioning_short_side,
            "reference_conditioning_max_width": options.reference_conditioning_max_width,
            "reference_conditioning_max_height": options.reference_conditioning_max_height,
        }
        if condition_height is not None and condition_width is not None:
            details.update(
                {
                    "condition_height": condition_height,
                    "condition_width": condition_width,
                    "condition_tokens": (condition_height // 16) * (condition_width // 16),
                }
            )
        emit("reference_vae_encode", started_at, **details)
        return result

    def profiled_create_conditioning(*args: Any, **kwargs: Any) -> tuple[Any, Any, int, int, int]:
        height = kwargs.get("height", args[1] if len(args) > 1 else None)
        width = kwargs.get("width", args[2] if len(args) > 2 else None)
        image_paths = kwargs.get("image_paths", args[3] if len(args) > 3 else None)
        vl_width = kwargs.get("vl_width", args[4] if len(args) > 4 else None)
        vl_height = kwargs.get("vl_height", args[5] if len(args) > 5 else None)
        override_dimensions = None
        if options.reference_conditioning_size != "upstream" and image_paths is not None:
            reference_path = normalize_image_paths(image_paths)[-1]
            reference_image = ImageUtil.load_image(reference_path).convert("RGB")
            image_width, image_height = reference_image.size
            override_dimensions = reference_conditioning_dimensions(
                policy=options.reference_conditioning_size,
                image_width=image_width,
                image_height=image_height,
                short_side=options.reference_conditioning_short_side,
                max_width=options.reference_conditioning_max_width,
                max_height=options.reference_conditioning_max_height,
            )
            if override_dimensions is not None:
                vl_width, vl_height = override_dimensions
                args, kwargs = replace_or_add_conditioning_size(
                    args,
                    kwargs,
                    width=vl_width,
                    height=vl_height,
                )
        if height is not None and width is not None and image_paths is not None:
            cache_key = conditioning_latents_cache_key(
                image_paths=image_paths,
                height=int(height),
                width=int(width),
                vl_width=vl_width,
                vl_height=vl_height,
            )
            cache_started_at = time.perf_counter()
            cached = conditioning_latents_cache.get(cache_key)
            if cached is not None:
                emit(
                    "conditioning_latents_cache_hit",
                    cache_started_at,
                    condition_height=cached[2] * 16,
                    condition_width=cached[3] * 16,
                    condition_tokens=cached[2] * cached[3] * cached[4],
                    reference_count=cached[4],
                    reference_conditioning_size=options.reference_conditioning_size,
                    reference_conditioning_short_side=options.reference_conditioning_short_side,
                    reference_conditioning_max_width=options.reference_conditioning_max_width,
                    reference_conditioning_max_height=options.reference_conditioning_max_height,
                )
                return cached
        else:
            cache_key = None

        started_at = time.perf_counter()
        result = original_create_conditioning(*args, **kwargs)
        mx.eval(result[0], result[1])
        emit(
            "conditioning_latents_total",
            started_at,
            condition_height=result[2] * 16,
            condition_width=result[3] * 16,
            condition_tokens=result[2] * result[3] * result[4],
            reference_count=result[4],
            reference_conditioning_size=options.reference_conditioning_size,
            reference_conditioning_short_side=options.reference_conditioning_short_side,
            reference_conditioning_max_width=options.reference_conditioning_max_width,
            reference_conditioning_max_height=options.reference_conditioning_max_height,
        )
        if cache_key is not None:
            conditioning_latents_cache[cache_key] = result
        return result

    def profiled_decode(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_decode(*args, **kwargs)
        mx.eval(result)
        emit("vae_decode", started_at)
        return result

    def profiled_to_image(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        result = original_to_image(*args, **kwargs)
        emit("image_conversion", started_at)
        return result

    def profiled_save(image: Any, *args: Any, **kwargs: Any) -> None:
        started_at = time.perf_counter()
        original_save(image, *args, **kwargs)
        emit("image_save", started_at)

    def materializing_block(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal block_zero_inputs, residual_cache_state
        nonlocal current_fn_hidden, current_fn_encoder, current_fn_residual
        nonlocal current_middle_residual, current_middle_encoder_residual
        nonlocal current_predicted_fn_residual, current_predicted_middle_residual
        nonlocal current_predicted_middle_encoder_residual
        nonlocal current_cache_hit, current_cache_reason, current_cache_relative_l1
        nonlocal current_cache_effective_threshold, current_cache_threshold_progress
        nonlocal current_cache_veto_threshold, current_cache_vetoed
        nonlocal current_cache_threshold_coordinate, current_cache_prediction_scale
        nonlocal current_total_steps, current_first_sigma, current_final_sigma
        nonlocal current_cache_prediction, current_cache_metrics, prediction_error_ema
        nonlocal current_target_token_count, current_target_grid_shape
        nonlocal current_cond_image_grid
        nonlocal current_text_token_count, current_valid_text_token_count
        nonlocal diagnose_previous_middle_residual, diagnose_previous_coordinate
        nonlocal diagnose_f1_uniqueness, selective_refill_indices, selective_refill_scales
        block_index = kwargs.get("idx", args[0] if args else None)
        if block_index is None:
            raise RuntimeError("mflux Transformer block call did not provide an index")
        block = kwargs.get("block", args[1] if len(args) > 1 else None)
        encoder_input = kwargs.get("encoder_hidden_states", args[3] if len(args) > 3 else None)
        hidden_input = kwargs.get("hidden_states", args[2] if len(args) > 2 else None)
        if encoder_input is None or hidden_input is None:
            raise RuntimeError("mflux Transformer block call did not provide hidden states")
        q6_profile_this_block = (
            options.q6_linear_profile
            and block is not None
            and current_step is not None
        )
        if q6_profile_this_block:
            q6_linear_profiler.register_block(block, int(block_index))

        skip_middle = (
            current_cache_enabled
            and current_cache_hit
            and current_middle_end_index is not None
            and 0 < block_index <= current_middle_end_index
        )
        q6_profile_context = (
            q6_linear_profiler.block_context(step=current_step, block=int(block_index) + 1)
            if q6_profile_this_block
            else nullcontext()
        )
        with q6_profile_context:
            if skip_middle:
                selective_outputs = maybe_run_selective_middle_refill(
                    block_index=block_index,
                    args=args,
                    kwargs=kwargs,
                    encoder_input=encoder_input,
                    hidden_input=hidden_input,
                )
                if selective_outputs is not None:
                    outputs = selective_outputs
                else:
                    middle_residual = (
                        current_predicted_middle_residual
                        if current_predicted_middle_residual is not None
                        else cached_middle_residual
                    )
                    middle_encoder_residual = (
                        current_predicted_middle_encoder_residual
                        if current_predicted_middle_encoder_residual is not None
                        else cached_middle_encoder_residual
                    )
                    if middle_residual is None or middle_encoder_residual is None:
                        raise RuntimeError("residual cache hit without a middle-block residual")
                    outputs = select_middle_anchor_outputs(
                        anchor_mode=options.cache_anchor_mode,
                        block_index=block_index,
                        encoder_input=encoder_input,
                        hidden_input=hidden_input,
                        cached_middle_encoder_anchor=middle_encoder_residual,
                        cached_middle_hidden_anchor=middle_residual,
                    )
                    if block_index == 1 and current_fn_hidden is not None:
                        dampened = maybe_apply_residual_dampen(
                            block_index=block_index,
                            encoder_bridged=outputs[0],
                            hidden_bridged=outputs[1],
                            hidden_f1=current_fn_hidden,
                        )
                        if dampened is not None:
                            outputs = dampened
            else:
                token_merged_outputs = maybe_run_token_merged_block(
                    block_index=block_index,
                    args=args,
                    kwargs=kwargs,
                    encoder_input=encoder_input,
                    hidden_input=hidden_input,
                )
                outputs = (
                    token_merged_outputs
                    if token_merged_outputs is not None
                    else original_block(*args, **kwargs)
                )

        if current_cache_enabled and block_index == 0:
            if options.cache_anchor_mode == "residual":
                current_fn_encoder = outputs[0]
                current_fn_hidden = outputs[1]
            current_fn_residual = outputs[1] - hidden_input
            current_cache_relative_l1 = None
            can_compare = (
                residual_cache_state.has_anchor
                and current_step is not None
                and current_step > options.cache_warmup_steps
                and residual_cache_state.consecutive_hits < options.cache_max_consecutive
            )
            decision_started_at = time.perf_counter()
            if can_compare:
                if (
                    cached_fn_residual is None
                    or cached_middle_residual is None
                    or cached_middle_encoder_residual is None
                ):
                    raise RuntimeError("residual cache anchor is incomplete")
                current_cache_prediction = select_predicted_anchor(
                    predictor=options.cache_predictor,
                    cached_anchor=cached_fn_residual,
                    previous_anchor=previous_cached_fn_residual,
                    older_anchor=older_cached_fn_residual,
                    older_coordinate=older_anchor_coordinate,
                    previous_coordinate=previous_anchor_coordinate,
                    anchor_coordinate=cached_anchor_coordinate,
                    current_coordinate=current_cache_coordinate,
                )
                current_predicted_fn_residual = current_cache_prediction.value
                current_cache_prediction_scale = current_cache_prediction.scale
                current_middle_prediction = select_predicted_anchor(
                    predictor=options.cache_predictor,
                    cached_anchor=cached_middle_residual,
                    previous_anchor=previous_cached_middle_residual,
                    older_anchor=older_cached_middle_residual,
                    older_coordinate=older_anchor_coordinate,
                    previous_coordinate=previous_anchor_coordinate,
                    anchor_coordinate=cached_anchor_coordinate,
                    current_coordinate=current_cache_coordinate,
                )
                current_predicted_middle_residual = current_middle_prediction.value
                current_middle_encoder_prediction = select_predicted_anchor(
                    predictor=options.cache_predictor,
                    cached_anchor=cached_middle_encoder_residual,
                    previous_anchor=previous_cached_middle_encoder_residual,
                    older_anchor=older_cached_middle_encoder_residual,
                    older_coordinate=older_anchor_coordinate,
                    previous_coordinate=previous_anchor_coordinate,
                    anchor_coordinate=cached_anchor_coordinate,
                    current_coordinate=current_cache_coordinate,
                )
                current_predicted_middle_encoder_residual = current_middle_encoder_prediction.value
                metrics = prediction_metrics(
                    current_predicted_fn_residual,
                    current_fn_residual,
                    target_token_count=current_target_token_count,
                )
                global_relative_l1 = metrics["relative_l1"]
                current_cache_relative_l1 = select_cache_decision_metric(
                    policy=options.cache_region_policy,
                    global_relative_l1=global_relative_l1,
                    target_relative_l1=metrics["target_relative_l1"],
                    condition_relative_l1=metrics["condition_relative_l1"],
                )
                current_cache_metrics = {
                    **metrics,
                    "global_relative_l1": global_relative_l1,
                    "decision_relative_l1": current_cache_relative_l1,
                }
            threshold_adjustment = cache_threshold_adjustment(
                options.cache_threshold,
                options.cache_threshold_schedule,
                step=current_step or 0,
                total_steps=current_total_steps,
                current_sigma=current_cache_coordinate
                if current_cache_coordinate_kind == "sigma"
                else None,
                first_sigma=current_first_sigma,
                final_sigma=current_final_sigma,
                prediction_cosine=current_cache_metrics.get("prediction_cosine"),
                magnitude_ratio=current_cache_metrics.get("prediction_magnitude_ratio"),
                history_relative_l1=prediction_error_ema,
            )
            current_cache_effective_threshold = threshold_adjustment.value
            current_cache_threshold_progress = threshold_adjustment.progress
            current_cache_threshold_coordinate = threshold_adjustment.coordinate
            current_cache_veto_threshold = threshold_adjustment.veto_threshold
            current_cache_hit, residual_cache_state, current_cache_reason = decide_residual_cache(
                residual_cache_state,
                step=current_step or 0,
                warmup_steps=options.cache_warmup_steps,
                threshold=current_cache_effective_threshold,
                max_consecutive=options.cache_max_consecutive,
                relative_l1=current_cache_relative_l1,
            )
            current_cache_vetoed, veto_reason = should_flow_veto_cache_hit(
                schedule=options.cache_threshold_schedule,
                cache_hit=current_cache_hit,
                relative_l1=current_cache_relative_l1,
                base_threshold=options.cache_threshold,
                veto_threshold=current_cache_veto_threshold,
                prediction_cosine=current_cache_metrics.get("prediction_cosine"),
                magnitude_ratio=current_cache_metrics.get("prediction_magnitude_ratio"),
            )
            if current_cache_vetoed:
                current_cache_hit = False
                residual_cache_state = ResidualCacheState(has_anchor=True)
                current_cache_reason = veto_reason or "flow_veto"
            if current_cache_metrics.get("global_relative_l1") is not None:
                current_error = float(current_cache_metrics["global_relative_l1"] or 0.0)
                prediction_error_ema = (
                    current_error
                    if prediction_error_ema is None
                    else 0.75 * prediction_error_ema + 0.25 * current_error
                )
            emit(
                "residual_cache_decision",
                decision_started_at,
                step=current_step,
                hit=current_cache_hit,
                reason=current_cache_reason,
                anchor_mode=options.cache_anchor_mode,
                relative_l1=(
                    rounded_optional(current_cache_relative_l1)
                    if current_cache_relative_l1 is not None
                    else None
                ),
                global_relative_l1=rounded_optional(current_cache_metrics.get("global_relative_l1")),
                target_relative_l1=rounded_optional(current_cache_metrics.get("target_relative_l1")),
                condition_relative_l1=rounded_optional(
                    current_cache_metrics.get("condition_relative_l1")
                ),
                prediction_abs_l1=rounded_optional(current_cache_metrics.get("prediction_abs_l1")),
                prediction_denom_l1=rounded_optional(
                    current_cache_metrics.get("prediction_denom_l1")
                ),
                prediction_cosine=rounded_optional(current_cache_metrics.get("prediction_cosine")),
                prediction_magnitude_ratio=rounded_optional(
                    current_cache_metrics.get("prediction_magnitude_ratio")
                ),
                prediction_error_ema=rounded_optional(prediction_error_ema),
                threshold=round(current_cache_effective_threshold, 8),
                flow_veto_threshold=rounded_optional(current_cache_veto_threshold),
                flow_vetoed=current_cache_vetoed,
                base_threshold=options.cache_threshold,
                threshold_schedule=options.cache_threshold_schedule,
                threshold_progress=(
                    round(current_cache_threshold_progress, 8)
                    if current_cache_threshold_progress is not None
                    else None
                ),
                threshold_coordinate=current_cache_threshold_coordinate,
                cache_predictor=options.cache_predictor,
                cache_region_policy=options.cache_region_policy,
                prediction_method=(
                    current_cache_prediction.method
                    if current_cache_prediction is not None
                    else None
                ),
                prediction_order=(
                    current_cache_prediction.order
                    if current_cache_prediction is not None
                    else None
                ),
                prediction_fallback=(
                    current_cache_prediction.fallback_reason
                    if current_cache_prediction is not None
                    else None
                ),
                prediction_scale=(
                    rounded_optional(current_cache_prediction_scale)
                    if current_cache_prediction_scale is not None
                    else None
                ),
                threshold_cosine_factor=rounded_optional(threshold_adjustment.cosine_factor),
                threshold_magnitude_factor=rounded_optional(threshold_adjustment.magnitude_factor),
                threshold_history_factor=rounded_optional(threshold_adjustment.history_factor),
                coordinate=current_cache_coordinate,
                coordinate_kind=current_cache_coordinate_kind,
            )
            if should_materialize_residual_anchor(
                block_index=block_index,
                cache_enabled=current_cache_enabled,
                cache_hit=current_cache_hit,
                middle_end_index=current_middle_end_index,
            ):
                materialize_started_at = time.perf_counter()
                if options.cache_anchor_mode == "absolute":
                    mx.eval(current_fn_residual)
                else:
                    mx.eval(current_fn_encoder, current_fn_hidden, current_fn_residual)
                emit(
                    "residual_anchor_materialize",
                    materialize_started_at,
                    step=current_step,
                    anchor="front",
                    block=block_index + 1,
                )

        # F1 uniqueness for phase-0 diagnose and phase-1A selective refill selection.
        if (
            block_index == 0
            and current_step is not None
            and (
                options.bridge_error_diagnose
                or options.selective_refill_fraction > 0.0
            )
        ):
            if current_fn_hidden is None:
                current_fn_encoder = outputs[0]
                current_fn_hidden = outputs[1]
            target_count = current_target_token_count
            if target_count is not None and target_count > 0:
                target_region = token_slice(current_fn_hidden, 0, target_count)
                if target_region is not None:
                    best = bipartite_best_match_similarities(target_region)
                    diagnose_f1_uniqueness = best
                    if best is not None and should_apply_selective_refill(
                        fraction=options.selective_refill_fraction,
                        current_step=current_step,
                        min_step=options.selective_refill_min_step,
                        cache_hit=current_cache_hit,
                    ):
                        if options.selective_refill_mode in (
                            "uniqueness-scale",
                            "uniqueness-boost",
                        ):
                            direction = (
                                "boost"
                                if options.selective_refill_mode == "uniqueness-boost"
                                else "dampen"
                            )
                            (
                                selective_refill_indices,
                                selective_refill_scales,
                            ) = uniqueness_scaled_residual_scales(
                                best,
                                fraction=options.selective_refill_fraction,
                                amount=options.selective_refill_dampen,
                                direction=direction,
                            )
                        else:
                            selective_refill_indices = select_unique_even_indices(
                                best,
                                fraction=options.selective_refill_fraction,
                            )
                            selective_refill_scales = None
                    else:
                        selective_refill_indices = None
                        selective_refill_scales = None

        if (
            current_cache_enabled
            and not current_cache_hit
            and current_middle_end_index is not None
            and block_index == current_middle_end_index
        ):
            if options.cache_anchor_mode == "absolute":
                current_middle_residual = outputs[1]
                current_middle_encoder_residual = outputs[0]
            else:
                if current_fn_hidden is None or current_fn_encoder is None:
                    raise RuntimeError("residual cache did not capture the F1 hidden state")
                current_middle_residual = outputs[1] - current_fn_hidden
                current_middle_encoder_residual = outputs[0] - current_fn_encoder
            if should_materialize_residual_anchor(
                block_index=block_index,
                cache_enabled=current_cache_enabled,
                cache_hit=current_cache_hit,
                middle_end_index=current_middle_end_index,
            ):
                materialize_started_at = time.perf_counter()
                mx.eval(current_middle_residual, current_middle_encoder_residual)
                emit(
                    "residual_anchor_materialize",
                    materialize_started_at,
                    step=current_step,
                    anchor="middle",
                    block=block_index + 1,
                )

        # Phase-0: full-pass bridge-error vs uniqueness (cache threshold must be 0).
        if (
            options.bridge_error_diagnose
            and current_middle_end_index is not None
            and block_index == current_middle_end_index
            and current_step is not None
            and current_fn_hidden is not None
        ):
            diagnose_started_at = time.perf_counter()
            if options.cache_anchor_mode == "absolute":
                actual_middle = outputs[1]
            else:
                actual_middle = outputs[1] - current_fn_hidden
            if (
                diagnose_previous_middle_residual is not None
                and diagnose_f1_uniqueness is not None
            ):
                emit_bridge_error_diagnosis(
                    predicted_middle=diagnose_previous_middle_residual,
                    actual_middle=actual_middle,
                    uniqueness=diagnose_f1_uniqueness,
                    block=block_index + 1,
                    started_at=diagnose_started_at,
                )
            mx.eval(actual_middle)
            diagnose_previous_middle_residual = actual_middle
            diagnose_previous_coordinate = current_cache_coordinate

        if options.probe_blocks and current_step is not None:
            if block_index == 0:
                block_zero_inputs = (
                    encoder_input,
                    hidden_input,
                )
            one_based_block = block_index + 1
            if one_based_block in options.probe_blocks:
                if block_zero_inputs is None:
                    raise RuntimeError("qwen-image-shardedit-mlx residual probe did not capture block-zero inputs")
                current_probe_residuals[one_based_block] = (
                    outputs[0] - block_zero_inputs[0],
                    outputs[1] - block_zero_inputs[1],
                )

        if options.token_redundancy_blocks and current_step is not None:
            one_based_block = block_index + 1
            if one_based_block in options.token_redundancy_blocks:
                redundancy_started_at = time.perf_counter()
                target_count = current_target_token_count
                total_tokens = hidden_input.shape[token_axis(hidden_input) or 0]
                emit_token_redundancy(
                    region_name="all",
                    region=hidden_input,
                    block=one_based_block,
                    started_at=redundancy_started_at,
                )
                if target_count is not None and 0 < target_count < total_tokens:
                    emit_token_redundancy(
                        region_name="target",
                        region=token_slice(hidden_input, 0, target_count),
                        block=one_based_block,
                        started_at=redundancy_started_at,
                        grid_shape=current_target_grid_shape,
                    )
                    emit_token_redundancy(
                        region_name="condition",
                        region=token_slice(hidden_input, target_count, total_tokens),
                        block=one_based_block,
                        started_at=redundancy_started_at,
                    )

        if should_materialize_block(block_index, options.eval_every_n_blocks):
            mx.eval(*outputs)
        return outputs

    def optimized_transformer_call(transformer: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_state, last_transformer_key, last_transformer_output
        nonlocal current_step, block_zero_inputs, previous_probe_residuals
        nonlocal cached_fn_residual, cached_middle_residual, cached_middle_encoder_residual
        nonlocal previous_cached_fn_residual, previous_cached_middle_residual
        nonlocal previous_cached_middle_encoder_residual
        nonlocal older_cached_fn_residual, older_cached_middle_residual
        nonlocal older_cached_middle_encoder_residual
        nonlocal cached_anchor_coordinate, previous_anchor_coordinate
        nonlocal older_anchor_coordinate
        nonlocal current_fn_hidden, current_fn_encoder, current_fn_residual
        nonlocal current_middle_residual, current_middle_encoder_residual
        nonlocal current_predicted_fn_residual, current_predicted_middle_residual
        nonlocal current_predicted_middle_encoder_residual
        nonlocal current_cache_enabled, current_cache_hit, current_cache_reason
        nonlocal current_cache_relative_l1, current_cache_effective_threshold
        nonlocal current_cache_threshold_progress, current_cache_threshold_coordinate
        nonlocal current_cache_veto_threshold, current_cache_vetoed
        nonlocal current_cache_prediction_scale, current_cache_coordinate
        nonlocal current_cache_prediction, current_cache_metrics
        nonlocal current_cache_coordinate_kind, current_total_steps
        nonlocal current_first_sigma, current_final_sigma
        nonlocal current_last_block_index, current_middle_end_index
        nonlocal current_target_token_count, current_target_grid_shape
        nonlocal current_cond_image_grid
        nonlocal current_text_token_count, current_valid_text_token_count
        nonlocal diagnose_f1_uniqueness, selective_refill_indices, selective_refill_scales
        timestep = kwargs.get("t", args[0] if args else None)
        config = kwargs.get("config", args[1] if len(args) > 1 else None)
        hidden_states = kwargs.get("hidden_states", args[2] if len(args) > 2 else None)
        encoder_hidden_states = kwargs.get(
            "encoder_hidden_states", args[3] if len(args) > 3 else None
        )
        encoder_hidden_states_mask = kwargs.get(
            "encoder_hidden_states_mask", args[4] if len(args) > 4 else None
        )
        cond_image_grid = kwargs.get(
            "cond_image_grid", args[6] if len(args) > 6 else None
        )
        if timestep is None or config is None or hidden_states is None:
            return original_transformer_call(transformer, *args, **kwargs)

        key = (id(config), int(timestep))
        should_run, call_state = decide_transformer_call(
            call_state,
            key,
            unit_guidance=config.guidance == 1.0,
        )
        if not should_run:
            if last_transformer_key != key or last_transformer_output is None:
                raise RuntimeError("unit-guidance Transformer output was not cached")
            return last_transformer_output

        started_at = time.perf_counter()
        current_step = int(timestep) + 1
        block_zero_inputs = None
        current_probe_residuals.clear()
        current_fn_hidden = None
        current_fn_encoder = None
        current_fn_residual = None
        current_middle_residual = None
        current_middle_encoder_residual = None
        current_predicted_fn_residual = None
        current_predicted_middle_residual = None
        current_predicted_middle_encoder_residual = None
        current_cache_enabled = options.cache_threshold > 0.0 and config.guidance == 1.0
        current_cache_hit = False
        current_cache_reason = "disabled"
        current_cache_relative_l1 = None
        current_cache_effective_threshold = options.cache_threshold
        current_cache_threshold_progress = None
        current_cache_threshold_coordinate = "fixed"
        current_cache_veto_threshold = None
        diagnose_f1_uniqueness = None
        selective_refill_indices = None
        selective_refill_scales = None
        current_cache_vetoed = False
        current_cache_prediction_scale = None
        current_cache_prediction = None
        current_cache_metrics = {}
        current_cache_coordinate, current_cache_coordinate_kind = cache_coordinate(
            config,
            current_step,
        )
        current_total_steps = int(getattr(config, "num_inference_steps", 0) or 0)
        current_first_sigma = scheduler_sigma(config, 0)
        current_final_sigma = scheduler_sigma(config, max(current_total_steps - 1, 0))
        current_last_block_index = (
            streaming_runtime.block_count - 1
            if streaming_runtime is not None
            else len(transformer.transformer_blocks) - 1
        )
        current_target_token_count = None
        current_target_grid_shape = None
        current_cond_image_grid = cond_image_grid
        current_text_token_count = None
        current_valid_text_token_count = None
        if options.text_token_merge and encoder_hidden_states is not None:
            encoder_shape = getattr(encoder_hidden_states, "shape", ())
            if len(encoder_shape) >= 3:
                current_text_token_count = int(encoder_shape[1])
                current_valid_text_token_count = uniform_valid_text_token_count(
                    encoder_hidden_states_mask,
                    current_text_token_count,
                )
        config_height = getattr(config, "height", None)
        config_width = getattr(config, "width", None)
        if config_height is not None and config_width is not None:
            grid_height = int(config_height) // 16
            grid_width = int(config_width) // 16
            current_target_token_count = grid_height * grid_width
            current_target_grid_shape = (grid_height, grid_width)
        if options.cache_back_blocks > current_last_block_index:
            raise RuntimeError("cache back blocks must be less than the Transformer block count")
        current_middle_end_index = current_last_block_index - options.cache_back_blocks

        def emit_residency_window(result: ResidencyWindowResult) -> None:
            emit_seconds(
                "residency_window",
                result.load_seconds
                + result.lora_seconds
                + result.prepare_seconds
                + result.compute_seconds
                + result.release_seconds,
                step=current_step,
                window=result.window_index + 1,
                blocks=f"{result.block_indices[0]}-{result.block_indices[-1]}",
                shards=list(result.shards),
                load_seconds=round(result.load_seconds, 6),
                lora_seconds=round(result.lora_seconds, 6),
                prepare_seconds=round(result.prepare_seconds, 6),
                compute_seconds=round(result.compute_seconds, 6),
                release_seconds=round(result.release_seconds, 6),
                release_policy=result.release_policy,
                lora_selected_keys=result.lora_selected_keys,
                lora_matched_keys=result.lora_matched_keys,
                lora_applied_layers=result.lora_applied_layers,
                lora_weight_cache_hits=result.lora_weight_cache_hits,
                lora_weight_cache_size=result.lora_weight_cache_size,
                patched_window_cache_hit=result.patched_window_cache_hit,
                patched_window_cache_size=result.patched_window_cache_size,
                kquant_img_ff_cache_hits=result.kquant_img_ff_cache_hits,
                kquant_img_ff_cache_misses=result.kquant_img_ff_cache_misses,
                kquant_img_ff_cache_size=result.kquant_img_ff_cache_size,
                kquant_img_ff_cache_bytes=result.kquant_img_ff_cache_bytes,
                active_after_compute_gib=round(result.active_after_compute_gib, 6),
                active_after_release_gib=round(result.active_after_release_gib, 6),
                mlx_peak_gib=round(result.peak_gib, 6),
            )

        def should_load_streaming_block(block_index: int) -> bool:
            if (
                not current_cache_enabled
                or not current_cache_hit
                or current_middle_end_index is None
            ):
                return True
            if (
                options.selective_refill_mode in SUBSET_REFILL_MODES
                and options.selective_refill_fraction > 0.0
                and selective_refill_indices is not None
                and block_index == current_middle_end_index
            ):
                return True
            return block_index > current_middle_end_index

        try:
            if streaming_runtime is None:
                result = original_transformer_call(transformer, *args, **kwargs)
            else:
                if encoder_hidden_states is None or encoder_hidden_states_mask is None:
                    raise RuntimeError(
                        "streaming Transformer call requires encoder hidden states and mask"
                    )
                result = streaming_runtime(
                    transformer,
                    t=int(timestep),
                    config=config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    cond_image_grid=cond_image_grid,
                    apply_block=QwenTransformer._apply_transformer_block,
                    should_load_block=should_load_streaming_block,
                    bridge_skipped_blocks=current_cache_enabled,
                    on_window=emit_residency_window,
                )
        finally:
            current_step = None
        # mflux otherwise lets each step retain a large deferred graph. Materializing
        # here makes the progress timing truthful and bounds cross-step memory growth.
        current_probe_arrays = [
            array
            for block in options.probe_blocks
            for array in current_probe_residuals.get(block, ())
        ]
        current_cache_arrays = []
        if current_cache_enabled and not current_cache_hit:
            if (
                current_fn_residual is None
                or current_middle_residual is None
                or current_middle_encoder_residual is None
            ):
                raise RuntimeError("full Transformer step did not produce residual cache tensors")
            current_cache_arrays = [
                current_fn_residual,
                current_middle_residual,
                current_middle_encoder_residual,
            ]
        mx.eval(result, *current_probe_arrays, *current_cache_arrays)
        if current_cache_arrays:
            older_cached_fn_residual = previous_cached_fn_residual
            older_cached_middle_residual = previous_cached_middle_residual
            older_cached_middle_encoder_residual = previous_cached_middle_encoder_residual
            older_anchor_coordinate = previous_anchor_coordinate
            previous_cached_fn_residual = cached_fn_residual
            previous_cached_middle_residual = cached_middle_residual
            previous_cached_middle_encoder_residual = cached_middle_encoder_residual
            previous_anchor_coordinate = cached_anchor_coordinate
            cached_fn_residual = current_fn_residual
            cached_middle_residual = current_middle_residual
            cached_middle_encoder_residual = current_middle_encoder_residual
            cached_anchor_coordinate = current_cache_coordinate
        emit(
            "denoise_transformer",
            started_at,
            step=int(timestep) + 1,
            cache_hit=current_cache_hit if current_cache_enabled else None,
            cache_reason=current_cache_reason if current_cache_enabled else None,
            blocks_executed=(
                1 + options.cache_back_blocks
                if current_cache_hit
                else current_last_block_index + 1
            ),
            cache_back_blocks=options.cache_back_blocks if current_cache_enabled else None,
        )
        if options.q6_linear_profile:
            for event in q6_linear_profiler.drain_step(
                step=int(timestep) + 1,
                cache_hit=current_cache_hit if current_cache_enabled else None,
                cache_reason=current_cache_reason if current_cache_enabled else None,
            ):
                emit_seconds("q6_linear_profile", event.seconds, **event.details())

        for block in options.probe_blocks:
            current = current_probe_residuals.get(block)
            if current is None:
                continue
            previous = previous_probe_residuals.get(block)
            if previous is None:
                emit(
                    "residual_probe",
                    time.perf_counter(),
                    step=int(timestep) + 1,
                    block=block,
                    has_previous=False,
                )
                continue

            probe_started_at = time.perf_counter()
            text_relative_l1 = mx.mean(mx.abs(previous[0] - current[0])) / mx.mean(
                mx.abs(previous[0])
            )
            image_relative_l1 = mx.mean(mx.abs(previous[1] - current[1])) / mx.mean(
                mx.abs(previous[1])
            )
            mx.eval(text_relative_l1, image_relative_l1)
            emit(
                "residual_probe",
                probe_started_at,
                step=int(timestep) + 1,
                block=block,
                has_previous=True,
                image_relative_l1=round(float(image_relative_l1.item()), 8),
                text_relative_l1=round(float(text_relative_l1.item()), 8),
            )
        previous_probe_residuals = dict(current_probe_residuals)
        last_transformer_key = key
        last_transformer_output = result
        return result

    QwenImage.compute_guided_noise = staticmethod(optimized)
    if options.q6_linear_profile:
        QuantizedLinear.__call__ = profiled_quantized_linear_call
    QwenTransformer.__call__ = optimized_transformer_call
    QwenTransformer._apply_transformer_block = staticmethod(materializing_block)
    QwenImageEdit.__init__ = profiled_model_init
    QwenImageEdit.generate_image = profiled_generate
    QwenImageEdit._encode_prompts_with_images = optimized_encode_prompts
    LatentCreator.encode_image = staticmethod(profiled_encode_image)
    QwenEditUtil.create_image_conditioning_latents = staticmethod(profiled_create_conditioning)
    VAEUtil.decode = staticmethod(profiled_decode)
    ImageUtil.to_image = staticmethod(profiled_to_image)
    GeneratedImage.save = profiled_save
    try:
        yield
    finally:
        QwenImage.compute_guided_noise = staticmethod(original_guidance)
        if options.q6_linear_profile:
            QuantizedLinear.__call__ = original_quantized_linear_call
        QwenTransformer.__call__ = original_transformer_call
        QwenTransformer._apply_transformer_block = staticmethod(original_block)
        QwenImageEdit.__init__ = original_model_init
        QwenImageEdit.generate_image = original_generate
        QwenImageEdit._encode_prompts_with_images = original_encode_prompts
        LatentCreator.encode_image = staticmethod(original_encode_image)
        QwenEditUtil.create_image_conditioning_latents = staticmethod(original_create_conditioning)
        VAEUtil.decode = staticmethod(original_decode)
        ImageUtil.to_image = staticmethod(original_to_image)
        GeneratedImage.save = original_save


def main() -> None:
    """Run the upstream mflux Qwen edit CLI with exact unit-CFG pruning."""

    options, upstream_args = parse_runtime_options(sys.argv[1:])
    try:
        from mflux.models.qwen.cli.qwen_image_edit_generate import main as mflux_main
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise SystemExit("mflux>=0.18 is required to run shardedit-mflux-edit") from exc

    original_argv = sys.argv
    sys.argv = [original_argv[0], *upstream_args]
    started_at = time.perf_counter()
    try:
        with install_runtime_overrides(options):
            mflux_main()
    finally:
        sys.argv = original_argv
    if options.profile:
        print(format_timing_event("process_total", time.perf_counter() - started_at), flush=True)


if __name__ == "__main__":
    main()
