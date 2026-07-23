"""Manifest-driven quality regression checks for qwen-image-shardedit-mlx image edits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from shardedit_mlx.quality_metrics import (
    ImageQualityMetrics,
    NormalizedBox,
    compare_image_crop_files,
    compare_image_files,
    metrics_pass_thresholds,
    normalized_box_from_raw,
)


@dataclass(frozen=True)
class QualityThresholds:
    max_mae: float | None = None
    min_psnr_db: float | None = None
    max_changed_channel_ratio: float | None = None


@dataclass(frozen=True)
class QualityCoverageRequirements:
    min_total_cases: int | None = None
    min_comparable_cases: int | None = None
    min_distinct_reference_images: int | None = None
    required_case_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityCase:
    id: str
    prompt: str
    baseline_output: str | None = None
    candidate_output: str | None = None
    reference_images: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    manual_checks: tuple[str, ...] = ()
    face_check: FaceCheck | None = None


@dataclass(frozen=True)
class FaceCheck:
    reference_image: str | None
    reference_box: NormalizedBox
    baseline_box: NormalizedBox
    candidate_box: NormalizedBox
    crop_size: int = 160


@dataclass(frozen=True)
class FaceQualityResult:
    reference_image: str
    crop_size: int
    reference_baseline: ImageQualityMetrics
    reference_candidate: ImageQualityMetrics
    baseline_candidate: ImageQualityMetrics


@dataclass(frozen=True)
class QualityManifest:
    thresholds: QualityThresholds
    coverage_requirements: QualityCoverageRequirements
    cases: tuple[QualityCase, ...]


@dataclass(frozen=True)
class QualityRegressionCoverage:
    total_cases: int
    comparable_cases: int
    missing_output_cases: int
    passed_cases: int
    failed_cases: int
    distinct_reference_images: int
    case_tags: tuple[str, ...]


@dataclass(frozen=True)
class QualityCaseResult:
    id: str
    status: str
    baseline_output: str | None
    candidate_output: str | None
    metrics: ImageQualityMetrics | None = None
    face_metrics: FaceQualityResult | None = None
    reason: str | None = None


@dataclass(frozen=True)
class QualityRegressionReport:
    passed: bool
    thresholds: QualityThresholds
    coverage_requirements: QualityCoverageRequirements
    coverage: QualityRegressionCoverage
    coverage_ready: bool
    coverage_blockers: tuple[str, ...]
    results: tuple[QualityCaseResult, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(raw: Any, key: str) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, int | float):
        raise ValueError(f"{key} must be a number")
    return float(raw)


def _thresholds(raw: dict[str, Any]) -> QualityThresholds:
    return QualityThresholds(
        max_mae=_optional_float(raw.get("max_mae"), "max_mae"),
        min_psnr_db=_optional_float(raw.get("min_psnr_db"), "min_psnr_db"),
        max_changed_channel_ratio=_optional_float(
            raw.get("max_changed_channel_ratio"),
            "max_changed_channel_ratio",
        ),
    )


def _optional_non_negative_int(raw: Any, key: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw


def _coverage_requirements(raw: dict[str, Any]) -> QualityCoverageRequirements:
    return QualityCoverageRequirements(
        min_total_cases=_optional_non_negative_int(
            raw.get("min_total_cases"),
            "min_total_cases",
        ),
        min_comparable_cases=_optional_non_negative_int(
            raw.get("min_comparable_cases"),
            "min_comparable_cases",
        ),
        min_distinct_reference_images=_optional_non_negative_int(
            raw.get("min_distinct_reference_images"),
            "min_distinct_reference_images",
        ),
        required_case_tags=_string_tuple(raw.get("required_case_tags"), "required_case_tags"),
    )


def _string_field(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"quality case {key} must be a non-empty string")
    return value


def _optional_string(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must contain non-empty strings")
    return tuple(value)


def _manual_checks(raw: dict[str, Any]) -> tuple[str, ...]:
    return _string_tuple(raw.get("manual_checks"), "manual_checks")


def _optional_positive_int(raw: Any, key: str, default: int) -> int:
    if raw is None:
        return default
    if not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{key} must be a positive integer")
    return raw


def _face_check(raw: dict[str, Any]) -> FaceCheck | None:
    value = raw.get("face_check")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("face_check must be an object")
    return FaceCheck(
        reference_image=_optional_string(value, "reference_image"),
        reference_box=normalized_box_from_raw(value.get("reference_box")),
        baseline_box=normalized_box_from_raw(value.get("baseline_box")),
        candidate_box=normalized_box_from_raw(value.get("candidate_box")),
        crop_size=_optional_positive_int(value.get("crop_size"), "face_check.crop_size", 160),
    )


def load_quality_manifest(data: dict[str, Any]) -> QualityManifest:
    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("quality manifest must contain a non-empty cases list")

    seen_ids: set[str] = set()
    cases: list[QualityCase] = []
    for raw in cases_raw:
        if not isinstance(raw, dict):
            raise ValueError("quality cases must be objects")
        case_id = _string_field(raw, "id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate quality case id: {case_id}")
        seen_ids.add(case_id)
        cases.append(
            QualityCase(
                id=case_id,
                prompt=_string_field(raw, "prompt"),
                baseline_output=_optional_string(
                    raw,
                    "baseline_output",
                    "current_baseline_output",
                ),
                candidate_output=_optional_string(
                    raw,
                    "candidate_output",
                    "current_candidate_output",
                ),
                reference_images=_string_tuple(raw.get("reference_images"), "reference_images"),
                tags=_string_tuple(raw.get("tags"), "tags"),
                manual_checks=_manual_checks(raw),
                face_check=_face_check(raw),
            )
        )

    return QualityManifest(
        thresholds=_thresholds(data.get("pixel_thresholds", {})),
        coverage_requirements=_coverage_requirements(data.get("coverage_requirements", {})),
        cases=tuple(cases),
    )


def load_quality_manifest_file(path: Path) -> QualityManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("quality manifest must be a JSON object")
    return load_quality_manifest(data)


def _resolve_output_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _missing_reason(reference: Path | None, candidate: Path | None) -> str | None:
    missing: list[str] = []
    if reference is None:
        missing.append("baseline_output")
    elif not reference.exists():
        missing.append(f"baseline_output:{reference}")
    if candidate is None:
        missing.append("candidate_output")
    elif not candidate.exists():
        missing.append(f"candidate_output:{candidate}")
    if not missing:
        return None
    return "missing " + ", ".join(missing)


def _face_quality_result(
    case: QualityCase,
    *,
    base_dir: Path,
    baseline_output: Path,
    candidate_output: Path,
) -> tuple[FaceQualityResult | None, str | None]:
    face_check = case.face_check
    if face_check is None:
        return None, None
    raw_reference = (
        face_check.reference_image
        if face_check.reference_image is not None
        else case.reference_images[0]
        if case.reference_images
        else None
    )
    if raw_reference is None:
        return None, "missing face reference_image"
    reference_path = _resolve_output_path(base_dir, raw_reference)
    if reference_path is None or not reference_path.exists():
        return None, f"missing face reference_image:{reference_path}"
    return (
        FaceQualityResult(
            reference_image=str(reference_path),
            crop_size=face_check.crop_size,
            reference_baseline=compare_image_crop_files(
                reference_path,
                face_check.reference_box,
                baseline_output,
                face_check.baseline_box,
                crop_size=face_check.crop_size,
            ),
            reference_candidate=compare_image_crop_files(
                reference_path,
                face_check.reference_box,
                candidate_output,
                face_check.candidate_box,
                crop_size=face_check.crop_size,
            ),
            baseline_candidate=compare_image_crop_files(
                baseline_output,
                face_check.baseline_box,
                candidate_output,
                face_check.candidate_box,
                crop_size=face_check.crop_size,
            ),
        ),
        None,
    )


def _coverage_blockers(
    coverage: QualityRegressionCoverage,
    requirements: QualityCoverageRequirements,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if (
        requirements.min_total_cases is not None
        and coverage.total_cases < requirements.min_total_cases
    ):
        blockers.append(
            f"total_cases {coverage.total_cases} < required {requirements.min_total_cases}"
        )
    if (
        requirements.min_comparable_cases is not None
        and coverage.comparable_cases < requirements.min_comparable_cases
    ):
        blockers.append(
            "comparable_cases "
            f"{coverage.comparable_cases} < required {requirements.min_comparable_cases}"
        )
    if (
        requirements.min_distinct_reference_images is not None
        and coverage.distinct_reference_images < requirements.min_distinct_reference_images
    ):
        blockers.append(
            "distinct_reference_images "
            f"{coverage.distinct_reference_images} < required "
            f"{requirements.min_distinct_reference_images}"
        )
    missing_tags = tuple(
        tag for tag in requirements.required_case_tags if tag not in coverage.case_tags
    )
    if missing_tags:
        blockers.append(f"missing required case tags: {', '.join(missing_tags)}")
    return tuple(blockers)


def evaluate_quality_manifest(
    manifest: QualityManifest,
    *,
    base_dir: Path,
    require_outputs: bool = False,
    require_coverage: bool = False,
) -> QualityRegressionReport:
    results: list[QualityCaseResult] = []
    for case in manifest.cases:
        reference = _resolve_output_path(base_dir, case.baseline_output)
        candidate = _resolve_output_path(base_dir, case.candidate_output)
        missing_reason = _missing_reason(reference, candidate)
        if missing_reason is not None:
            results.append(
                QualityCaseResult(
                    id=case.id,
                    status="missing_outputs",
                    baseline_output=str(reference) if reference is not None else None,
                    candidate_output=str(candidate) if candidate is not None else None,
                    reason=missing_reason,
                )
            )
            continue

        assert reference is not None
        assert candidate is not None
        metrics = compare_image_files(reference, candidate)
        face_metrics, face_reason = _face_quality_result(
            case,
            base_dir=base_dir,
            baseline_output=reference,
            candidate_output=candidate,
        )
        passed = metrics_pass_thresholds(
            metrics,
            max_mae=manifest.thresholds.max_mae,
            min_psnr_db=manifest.thresholds.min_psnr_db,
            max_changed_channel_ratio=manifest.thresholds.max_changed_channel_ratio,
        ) and face_reason is None
        results.append(
            QualityCaseResult(
                id=case.id,
                status="passed" if passed else "failed",
                baseline_output=str(reference),
                candidate_output=str(candidate),
                metrics=metrics,
                face_metrics=face_metrics,
                reason=face_reason,
            )
        )

    missing_count = sum(1 for result in results if result.status == "missing_outputs")
    failed_count = sum(1 for result in results if result.status == "failed")
    passed_count = sum(1 for result in results if result.status == "passed")
    coverage = QualityRegressionCoverage(
        total_cases=len(results),
        comparable_cases=passed_count + failed_count,
        missing_output_cases=missing_count,
        passed_cases=passed_count,
        failed_cases=failed_count,
        distinct_reference_images=len(
            {image for case in manifest.cases for image in case.reference_images}
        ),
        case_tags=tuple(sorted({tag for case in manifest.cases for tag in case.tags})),
    )
    blockers = _coverage_blockers(coverage, manifest.coverage_requirements)
    coverage_ready = not blockers
    return QualityRegressionReport(
        passed=(
            failed_count == 0
            and (not require_outputs or missing_count == 0)
            and (not require_coverage or coverage_ready)
        ),
        thresholds=manifest.thresholds,
        coverage_requirements=manifest.coverage_requirements,
        coverage=coverage,
        coverage_ready=coverage_ready,
        coverage_blockers=blockers,
        results=tuple(results),
    )


def evaluate_quality_manifest_file(
    path: Path,
    *,
    base_dir: Path | None = None,
    require_outputs: bool = False,
    require_coverage: bool = False,
) -> QualityRegressionReport:
    return evaluate_quality_manifest(
        load_quality_manifest_file(path),
        base_dir=Path.cwd() if base_dir is None else base_dir,
        require_outputs=require_outputs,
        require_coverage=require_coverage,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("benchmarks/quality_cases.json"),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative output paths",
    )
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Fail if any manifest case is missing generated outputs",
    )
    parser.add_argument(
        "--require-coverage",
        action="store_true",
        help="Fail if manifest coverage requirements are not met",
    )
    args = parser.parse_args()

    report = evaluate_quality_manifest_file(
        args.manifest,
        base_dir=args.base_dir,
        require_outputs=args.require_outputs,
        require_coverage=args.require_coverage,
    )
    print(json.dumps(report.to_json_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
