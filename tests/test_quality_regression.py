from __future__ import annotations

import json

import pytest

from shardedit_mlx.quality_regression import (
    evaluate_quality_manifest,
    evaluate_quality_manifest_file,
    load_quality_manifest,
    load_quality_manifest_file,
)


def _write_rgb(path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (1, 1), color).save(path)


def _write_manifest(path, cases: list[dict], thresholds: dict | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "last_updated": "2026-07-19",
                "pixel_thresholds": thresholds or {"max_mae": 2.0},
                "cases": cases,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_quality_manifest_reports_reference_and_tag_coverage(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _write_rgb(reference, (0, 0, 0))
    _write_rgb(candidate, (1, 0, 0))
    manifest = load_quality_manifest(
        {
            "pixel_thresholds": {"max_mae": 2.0},
            "coverage_requirements": {
                "min_total_cases": 2,
                "min_comparable_cases": 1,
                "min_distinct_reference_images": 2,
                "required_case_tags": ["portrait", "lighting"],
            },
            "cases": [
                {
                    "id": "same_ref_a",
                    "prompt": "keep identity in cafe",
                    "reference_images": ["person-a.png"],
                    "tags": ["portrait", "lighting"],
                    "baseline_output": str(reference),
                    "candidate_output": str(candidate),
                },
                {
                    "id": "same_ref_b",
                    "prompt": "keep identity in studio",
                    "reference_images": ["person-a.png"],
                    "tags": ["portrait"],
                },
            ],
        }
    )

    report = evaluate_quality_manifest(manifest, base_dir=tmp_path)

    assert report.coverage.total_cases == 2
    assert report.coverage.comparable_cases == 1
    assert report.coverage.distinct_reference_images == 1
    assert report.coverage.case_tags == ("lighting", "portrait")
    assert not report.coverage_ready
    assert report.coverage_blockers == (
        "distinct_reference_images 1 < required 2",
    )


def test_quality_manifest_can_require_promotion_coverage(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _write_rgb(reference, (0, 0, 0))
    _write_rgb(candidate, (1, 0, 0))
    manifest = load_quality_manifest(
        {
            "pixel_thresholds": {"max_mae": 2.0},
            "coverage_requirements": {
                "min_total_cases": 2,
                "min_comparable_cases": 2,
                "min_distinct_reference_images": 2,
                "required_case_tags": ["portrait", "pose"],
            },
            "cases": [
                {
                    "id": "only_case",
                    "prompt": "keep identity",
                    "reference_images": ["person-a.png"],
                    "tags": ["portrait"],
                    "baseline_output": str(reference),
                    "candidate_output": str(candidate),
                }
            ],
        }
    )

    report = evaluate_quality_manifest(
        manifest,
        base_dir=tmp_path,
        require_coverage=True,
    )

    assert not report.passed
    assert not report.coverage_ready
    assert report.coverage_blockers == (
        "total_cases 1 < required 2",
        "comparable_cases 1 < required 2",
        "distinct_reference_images 1 < required 2",
        "missing required case tags: pose",
    )


def test_quality_manifest_evaluates_comparable_cases_and_reports_missing_outputs(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    manifest = tmp_path / "quality.json"
    _write_rgb(reference, (0, 0, 0))
    _write_rgb(candidate, (3, 0, 0))
    _write_manifest(
        manifest,
        [
            {
                "id": "measured",
                "prompt": "keep identity",
                "current_baseline_output": "reference.png",
                "current_candidate_output": "candidate.png",
            },
            {"id": "missing", "prompt": "needs generation"},
        ],
    )

    report = evaluate_quality_manifest_file(manifest, base_dir=tmp_path)

    assert report.passed
    assert report.coverage.total_cases == 2
    assert report.coverage.comparable_cases == 1
    assert report.coverage.missing_output_cases == 1
    assert report.results[0].status == "passed"
    assert report.results[0].metrics is not None
    assert report.results[0].metrics.mae == pytest.approx(1.0)
    assert report.results[1].status == "missing_outputs"


def test_quality_manifest_reports_reference_vs_output_face_metrics(tmp_path) -> None:
    from PIL import Image

    reference_image = tmp_path / "person.png"
    baseline = tmp_path / "baseline.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(reference_image)
    Image.new("RGB", (4, 4), (3, 0, 0)).save(baseline)
    Image.new("RGB", (4, 4), (6, 0, 0)).save(candidate)
    manifest = load_quality_manifest(
        {
            "pixel_thresholds": {"max_mae": 4.0},
            "cases": [
                {
                    "id": "face_checked",
                    "prompt": "keep identity",
                    "reference_images": ["person.png"],
                    "baseline_output": "baseline.png",
                    "candidate_output": "candidate.png",
                    "face_check": {
                        "reference_box": [0.25, 0.25, 0.75, 0.75],
                        "baseline_box": [0.25, 0.25, 0.75, 0.75],
                        "candidate_box": [0.25, 0.25, 0.75, 0.75],
                        "crop_size": 2,
                    },
                }
            ],
        }
    )

    report = evaluate_quality_manifest(manifest, base_dir=tmp_path)

    assert report.passed
    face_metrics = report.results[0].face_metrics
    assert face_metrics is not None
    assert face_metrics.reference_baseline.mae == pytest.approx(1.0)
    assert face_metrics.reference_candidate.mae == pytest.approx(2.0)
    assert face_metrics.baseline_candidate.mae == pytest.approx(1.0)


def test_quality_manifest_fails_when_configured_face_reference_is_missing(tmp_path) -> None:
    baseline = tmp_path / "baseline.png"
    candidate = tmp_path / "candidate.png"
    _write_rgb(baseline, (0, 0, 0))
    _write_rgb(candidate, (0, 0, 0))
    manifest = load_quality_manifest(
        {
            "pixel_thresholds": {"max_mae": 0.0},
            "cases": [
                {
                    "id": "missing_face_ref",
                    "prompt": "keep identity",
                    "baseline_output": "baseline.png",
                    "candidate_output": "candidate.png",
                    "face_check": {
                        "reference_image": "person.png",
                        "reference_box": [0.0, 0.0, 1.0, 1.0],
                        "baseline_box": [0.0, 0.0, 1.0, 1.0],
                        "candidate_box": [0.0, 0.0, 1.0, 1.0],
                    },
                }
            ],
        }
    )

    report = evaluate_quality_manifest(manifest, base_dir=tmp_path)

    assert not report.passed
    assert report.results[0].status == "failed"
    assert report.results[0].reason is not None
    assert "missing face reference_image" in report.results[0].reason


def test_quality_manifest_can_require_outputs_for_every_case(tmp_path) -> None:
    manifest = tmp_path / "quality.json"
    _write_manifest(manifest, [{"id": "missing", "prompt": "needs generation"}])

    report = evaluate_quality_manifest_file(
        manifest,
        base_dir=tmp_path,
        require_outputs=True,
    )

    assert not report.passed
    assert report.results[0].status == "missing_outputs"


def test_quality_manifest_fails_when_pixel_thresholds_fail(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    manifest = tmp_path / "quality.json"
    _write_rgb(reference, (0, 0, 0))
    _write_rgb(candidate, (9, 0, 0))
    _write_manifest(
        manifest,
        [
            {
                "id": "softened",
                "prompt": "keep identity",
                "baseline_output": "reference.png",
                "candidate_output": "candidate.png",
            }
        ],
        thresholds={"max_mae": 2.0},
    )

    report = evaluate_quality_manifest_file(manifest, base_dir=tmp_path)

    assert not report.passed
    assert report.coverage.failed_cases == 1
    assert report.results[0].status == "failed"


def test_quality_manifest_rejects_duplicate_case_ids(tmp_path) -> None:
    manifest = tmp_path / "quality.json"
    _write_manifest(
        manifest,
        [
            {"id": "duplicate", "prompt": "first"},
            {"id": "duplicate", "prompt": "second"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate quality case id"):
        load_quality_manifest_file(manifest)
