"""Pixel-level quality metrics for qwen-image-shardedit-mlx image A/B checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageQualityMetrics:
    mae: float
    rmse: float
    psnr_db: float
    max_abs: int
    changed_channel_ratio: float
    channel_count: int


NormalizedBox = tuple[float, float, float, float]


def normalized_box_from_sequence(raw: Sequence[float]) -> NormalizedBox:
    """Validate a normalized left/top/right/bottom crop box."""

    if len(raw) != 4:
        raise ValueError("crop box must contain four values")
    try:
        left, top, right, bottom = (float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("crop box values must be numbers") from exc
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("crop box values must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return left, top, right, bottom


def normalized_box_from_raw(raw: Any) -> NormalizedBox:
    """Parse a normalized box from either a list or a dict."""

    if isinstance(raw, dict):
        return normalized_box_from_sequence(
            (
                raw.get("left"),
                raw.get("top"),
                raw.get("right"),
                raw.get("bottom"),
            )
        )
    if isinstance(raw, list | tuple):
        return normalized_box_from_sequence(raw)
    raise ValueError("crop box must be a list or object")


def compute_channel_metrics(
    reference_channels: Sequence[int],
    candidate_channels: Sequence[int],
    *,
    max_channel_value: int = 255,
) -> ImageQualityMetrics:
    """Compare same-length channel streams."""

    if len(reference_channels) != len(candidate_channels):
        raise ValueError("reference and candidate channel streams must have the same length")
    if not reference_channels:
        raise ValueError("channel streams cannot be empty")

    absolute_errors = tuple(
        abs(int(reference) - int(candidate))
        for reference, candidate in zip(reference_channels, candidate_channels, strict=True)
    )
    squared_error_sum = sum(error * error for error in absolute_errors)
    channel_count = len(absolute_errors)
    mae = sum(absolute_errors) / channel_count
    mse = squared_error_sum / channel_count
    rmse = math.sqrt(mse)
    psnr_db = math.inf if mse == 0 else 20 * math.log10(max_channel_value / rmse)
    changed = sum(1 for error in absolute_errors if error != 0)
    return ImageQualityMetrics(
        mae=mae,
        rmse=rmse,
        psnr_db=psnr_db,
        max_abs=max(absolute_errors),
        changed_channel_ratio=changed / channel_count,
        channel_count=channel_count,
    )


def _crop_channels(path: Path, box: NormalizedBox, crop_size: int) -> tuple[int, ...]:
    from PIL import Image

    if crop_size < 1:
        raise ValueError("crop size must be >= 1")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        left, top, right, bottom = box
        crop = rgb.crop(
            (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
        )
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        resized = crop.resize((crop_size, crop_size), resampling)
        return tuple(resized.tobytes())


def compare_image_crop_files(
    reference_path: Path,
    reference_box: NormalizedBox,
    candidate_path: Path,
    candidate_box: NormalizedBox,
    *,
    crop_size: int = 160,
) -> ImageQualityMetrics:
    """Compare normalized crops from two images after resizing to a fixed square."""

    reference_channels = _crop_channels(reference_path, reference_box, crop_size)
    candidate_channels = _crop_channels(candidate_path, candidate_box, crop_size)
    return compute_channel_metrics(reference_channels, candidate_channels)


def compare_image_files(reference_path: Path, candidate_path: Path) -> ImageQualityMetrics:
    """Load two same-size image files and compare RGB channels."""

    from PIL import Image

    with Image.open(reference_path) as reference_image:
        reference = reference_image.convert("RGB")
        reference_size = reference.size
        reference_channels = tuple(reference.tobytes())
    with Image.open(candidate_path) as candidate_image:
        candidate = candidate_image.convert("RGB")
        if candidate.size != reference_size:
            raise ValueError(
                f"image sizes differ: reference={reference_size}, candidate={candidate.size}"
            )
        candidate_channels = tuple(candidate.tobytes())
    return compute_channel_metrics(reference_channels, candidate_channels)


def metrics_pass_thresholds(
    metrics: ImageQualityMetrics,
    *,
    max_mae: float | None = None,
    min_psnr_db: float | None = None,
    max_changed_channel_ratio: float | None = None,
) -> bool:
    if max_mae is not None and metrics.mae > max_mae:
        return False
    if min_psnr_db is not None and metrics.psnr_db < min_psnr_db:
        return False
    if (
        max_changed_channel_ratio is not None
        and metrics.changed_channel_ratio > max_changed_channel_ratio
    ):
        return False
    return True
