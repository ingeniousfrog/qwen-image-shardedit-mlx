from __future__ import annotations

import math

import pytest

from shardedit_mlx.quality_metrics import (
    compare_image_crop_files,
    compare_image_files,
    compute_channel_metrics,
    metrics_pass_thresholds,
    normalized_box_from_raw,
)


def test_identical_channels_have_zero_error_and_infinite_psnr() -> None:
    metrics = compute_channel_metrics((0, 128, 255), (0, 128, 255))

    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert math.isinf(metrics.psnr_db)
    assert metrics.max_abs == 0
    assert metrics.changed_channel_ratio == 0


def test_channel_metrics_report_absolute_and_squared_error() -> None:
    metrics = compute_channel_metrics((0, 10, 20), (10, 10, 50))

    assert metrics.mae == pytest.approx(40 / 3)
    assert metrics.rmse == pytest.approx(math.sqrt((100 + 0 + 900) / 3))
    assert metrics.max_abs == 30
    assert metrics.changed_channel_ratio == pytest.approx(2 / 3)
    assert metrics.psnr_db < 30


def test_channel_metrics_reject_mismatched_or_empty_streams() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_channel_metrics((1,), (1, 2))
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_channel_metrics((), ())


def test_threshold_gate_combines_optional_limits() -> None:
    metrics = compute_channel_metrics((0, 10, 20), (10, 10, 50))

    assert metrics_pass_thresholds(metrics, max_mae=14, min_psnr_db=20)
    assert not metrics_pass_thresholds(metrics, max_mae=10)
    assert not metrics_pass_thresholds(metrics, min_psnr_db=40)
    assert not metrics_pass_thresholds(metrics, max_changed_channel_ratio=0.5)


def test_compare_image_files_requires_matching_sizes(tmp_path) -> None:
    from PIL import Image

    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (1, 1), (0, 0, 0)).save(reference)
    Image.new("RGB", (2, 1), (0, 0, 0)).save(candidate)

    with pytest.raises(ValueError, match="image sizes differ"):
        compare_image_files(reference, candidate)


def test_compare_image_crop_files_resizes_normalized_crops(tmp_path) -> None:
    from PIL import Image

    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
    Image.new("RGB", (8, 8), (10, 0, 0)).save(candidate)

    metrics = compare_image_crop_files(
        reference,
        (0.25, 0.25, 0.75, 0.75),
        candidate,
        (0.25, 0.25, 0.75, 0.75),
        crop_size=2,
    )

    assert metrics.mae == pytest.approx(10 / 3)
    assert metrics.channel_count == 12


def test_normalized_box_from_raw_accepts_dict_and_rejects_invalid_box() -> None:
    assert normalized_box_from_raw(
        {"left": 0.1, "top": 0.2, "right": 0.8, "bottom": 0.9}
    ) == (0.1, 0.2, 0.8, 0.9)
    with pytest.raises(ValueError, match="left < right"):
        normalized_box_from_raw([0.9, 0.1, 0.8, 0.9])
