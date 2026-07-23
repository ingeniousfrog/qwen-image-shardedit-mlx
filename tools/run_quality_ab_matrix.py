#!/usr/bin/env python3
"""Run the qwen-image-shardedit-mlx 6-case quality A/B benchmark matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    id: str
    description: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class MatrixRecord:
    case_id: str
    variant_id: str
    status: int
    run_dir: str | None
    output: str | None
    command: tuple[str, ...]


VARIANTS: dict[str, Variant] = {
    "upstream_nocache": Variant(
        id="upstream_nocache",
        description="upstream reference conditioning no-cache shard baseline",
        args=(
            "--reference-conditioning-size",
            "upstream",
            "--cache-threshold",
            "0",
        ),
    ),
    "upstream_f1b2": Variant(
        id="upstream_f1b2",
        description="upstream reference conditioning F1B2 fixed-threshold candidate",
        args=(
            "--reference-conditioning-size",
            "upstream",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
        ),
    ),
    "upstream_f1b2_condmerge": Variant(
        id="upstream_f1b2_condmerge",
        description="rejected diagnostic: upstream F1B2 plus condition-only token merge V0",
        args=(
            "--reference-conditioning-size",
            "upstream",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--condition-token-merge",
            "--condition-token-merge-stride",
            "2",
            "--condition-token-merge-start-block",
            "2",
            "--condition-token-merge-back-blocks",
            "2",
        ),
    ),
    "fitbox_nocache": Variant(
        id="fitbox_nocache",
        description="controlled cond576 fit-box no-cache shard baseline",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0",
        ),
    ),
    "fitbox_f1b2": Variant(
        id="fitbox_f1b2",
        description="controlled cond576 fit-box F1B2 fixed-threshold candidate",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
        ),
    ),
    "fitbox_f1b2_condmerge": Variant(
        id="fitbox_f1b2_condmerge",
        description=(
            "rejected diagnostic: controlled cond576 fit-box F1B2 plus "
            "condition-only token merge V0"
        ),
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--condition-token-merge",
            "--condition-token-merge-stride",
            "2",
            "--condition-token-merge-start-block",
            "2",
            "--condition-token-merge-back-blocks",
            "2",
        ),
    ),
    "fitbox_f1b2_textmerge": Variant(
        id="fitbox_f1b2_textmerge",
        description=(
            "diagnostic: controlled cond576 fit-box F1B2 plus text-only "
            "token merge V0, no smoke speedup"
        ),
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--text-token-merge",
            "--text-token-merge-stride",
            "2",
            "--text-token-merge-start-block",
            "2",
            "--text-token-merge-back-blocks",
            "2",
        ),
    ),
    "fitbox_f1b2_bothmerge": Variant(
        id="fitbox_f1b2_bothmerge",
        description=(
            "rejected diagnostic: controlled cond576 fit-box F1B2 plus "
            "text and condition token merge V0"
        ),
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--text-token-merge",
            "--text-token-merge-stride",
            "2",
            "--text-token-merge-start-block",
            "2",
            "--text-token-merge-back-blocks",
            "2",
            "--condition-token-merge",
            "--condition-token-merge-stride",
            "2",
            "--condition-token-merge-start-block",
            "2",
            "--condition-token-merge-back-blocks",
            "2",
        ),
    ),
    "fitbox_f1b2_flowaware": Variant(
        id="fitbox_f1b2_flowaware",
        description="controlled cond576 fit-box F1B2 flow-aware threshold candidate",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-preset",
            "flow-aware",
        ),
    ),
    "fitbox_f1b2_flowveto": Variant(
        id="fitbox_f1b2_flowveto",
        description="controlled cond576 fit-box F1B2 fixed cadence with flow-aware veto",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-preset",
            "flow-aware-veto",
        ),
    ),
    "fitbox_f1b2_linear": Variant(
        id="fitbox_f1b2_linear",
        description="controlled cond576 fit-box F1B2 fixed-threshold linear predictor candidate",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-preset",
            "f1b2-linear",
        ),
    ),
    "fitbox_f1b2_ab2": Variant(
        id="fitbox_f1b2_ab2",
        description="controlled cond576 fit-box F1B2 fixed-threshold Adams-Bashforth predictor candidate",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-preset",
            "f1b2-ab2",
        ),
    ),
    "short512_nocache": Variant(
        id="short512_nocache",
        description="short-side-512 no-cache shard baseline",
        args=(
            "--reference-conditioning-size",
            "short-side-512",
            "--cache-threshold",
            "0",
        ),
    ),
    "short512_f1b2": Variant(
        id="short512_f1b2",
        description="short-side-512 F1B2 fixed-threshold candidate",
        args=(
            "--reference-conditioning-size",
            "short-side-512",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
        ),
    ),
    "short512_f1b2_condmerge": Variant(
        id="short512_f1b2_condmerge",
        description="rejected diagnostic: short-side-512 F1B2 plus condition-only token merge V0",
        args=(
            "--reference-conditioning-size",
            "short-side-512",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--condition-token-merge",
            "--condition-token-merge-stride",
            "2",
            "--condition-token-merge-start-block",
            "2",
            "--condition-token-merge-back-blocks",
            "2",
        ),
    ),
    "fitbox_taylor_flowaware": Variant(
        id="fitbox_taylor_flowaware",
        description="controlled cond576 fit-box quadratic predictor with flow-aware threshold",
        args=(
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--cache-predictor",
            "quadratic",
            "--cache-threshold-schedule",
            "flow-aware",
            "--cache-region-policy",
            "target-conservative",
        ),
    ),
    "short512_taylor_flowaware": Variant(
        id="short512_taylor_flowaware",
        description="short-side-512 quadratic predictor with flow-aware threshold",
        args=(
            "--reference-conditioning-size",
            "short-side-512",
            "--cache-threshold",
            "0.8",
            "--cache-max-consecutive",
            "1",
            "--cache-back-blocks",
            "2",
            "--cache-predictor",
            "quadratic",
            "--cache-threshold-schedule",
            "flow-aware",
            "--cache-region-policy",
            "target-conservative",
        ),
    ),
}


DEFAULT_VARIANTS = (
    "fitbox_nocache",
    "fitbox_f1b2",
    "short512_nocache",
    "short512_f1b2",
)

DIAGNOSTIC_CONDITION_MERGE_V0 = (
    "upstream_nocache",
    "upstream_f1b2",
    "upstream_f1b2_condmerge",
    "short512_nocache",
    "short512_f1b2",
    "short512_f1b2_condmerge",
    "fitbox_nocache",
    "fitbox_f1b2",
    "fitbox_f1b2_condmerge",
)

DIAGNOSTIC_TOKEN_MERGE_V0 = (
    "fitbox_f1b2",
    "fitbox_f1b2_textmerge",
    "fitbox_f1b2_condmerge",
    "fitbox_f1b2_bothmerge",
)

VARIANT_SETS = {
    "default": DEFAULT_VARIANTS,
    "cond576-flowaware": (
        "fitbox_nocache",
        "fitbox_f1b2",
        "fitbox_f1b2_flowaware",
    ),
    "cond576-acceleration": (
        "fitbox_nocache",
        "fitbox_f1b2",
        "fitbox_f1b2_flowaware",
        "fitbox_f1b2_flowveto",
        "fitbox_f1b2_linear",
        "fitbox_f1b2_ab2",
        "fitbox_taylor_flowaware",
    ),
    "cond576-next-cache": (
        "fitbox_nocache",
        "fitbox_f1b2",
        "fitbox_f1b2_flowaware",
        "fitbox_f1b2_flowveto",
        "fitbox_f1b2_linear",
        "fitbox_f1b2_ab2",
    ),
    "diagnostic-condition-merge-v0": DIAGNOSTIC_CONDITION_MERGE_V0,
    "condition-merge-v0": DIAGNOSTIC_CONDITION_MERGE_V0,
    "diagnostic-token-merge-v0": DIAGNOSTIC_TOKEN_MERGE_V0,
    "fitbox-token-merge-v0": DIAGNOSTIC_TOKEN_MERGE_V0,
    "all": tuple(VARIANTS),
}


def _load_cases(manifest_path: Path) -> list[dict]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain cases")
    return cases


def _case_reference(case: dict) -> str:
    images = case.get("reference_images")
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise ValueError(f"case {case.get('id')} must contain at least one reference image")
    return images[0]


def _parse_run_dir(output: str) -> str | None:
    for line in output.splitlines():
        if line.startswith("benchmark directory: "):
            return line.removeprefix("benchmark directory: ").strip()
    return None


def _output_path(run_dir: str | None) -> str | None:
    if run_dir is None:
        return None
    output = REPO_ROOT / run_dir / "shardedit-1.png"
    return str(output) if output.exists() else None


def _report_paths(output_root: Path, label: str) -> tuple[Path, Path]:
    date_path = time.strftime("%Y-%m-%d")
    report_dir = output_root / date_path
    report_dir.mkdir(parents=True, exist_ok=True)
    return (
        report_dir / f"{label}.jsonl",
        report_dir / f"{label}.json",
    )


def _write_jsonl(path: Path, record: MatrixRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _write_json(path: Path, records: list[MatrixRecord]) -> None:
    path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _record_from_json(raw: dict, line_number: int) -> MatrixRecord:
    try:
        command = raw["command"]
        if not isinstance(command, list):
            raise ValueError("command must be a list")
        record = MatrixRecord(
            case_id=str(raw["case_id"]),
            variant_id=str(raw["variant_id"]),
            status=int(raw["status"]),
            run_dir=None if raw.get("run_dir") is None else str(raw["run_dir"]),
            output=None if raw.get("output") is None else str(raw["output"]),
            command=tuple(str(part) for part in command),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid resume record on line {line_number}") from exc
    return record


def _load_jsonl_records(path: Path) -> list[MatrixRecord]:
    records: list[MatrixRecord] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"invalid resume record on line {line_number}")
            records.append(_record_from_json(raw, line_number))
    return records


def _completed_keys(records: list[MatrixRecord]) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    for record in records:
        if record.status == 0 and record.output and Path(record.output).exists():
            completed.add((record.case_id, record.variant_id))
    return completed


def _run_command(command: list[str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    status = process.wait()
    return status, "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "benchmarks/quality_cases.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmark-runs")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--cooldown-seconds", type=int, default=0)
    parser.add_argument(
        "--variant-set",
        choices=tuple(VARIANT_SETS),
        default="default",
        help=f"Named variant set. Available: {', '.join(VARIANT_SETS)}",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help=f"Optional comma-separated variants overriding --variant-set. Available: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--cases", default="", help="Optional comma-separated case ids")
    parser.add_argument("--label", default="", help="Report label; defaults to quality-ab-matrix timestamp")
    parser.add_argument(
        "--resume-report",
        type=Path,
        default=None,
        help="Existing JSONL matrix report to append to while skipping completed rows",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.variants:
        variant_ids = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    else:
        variant_ids = VARIANT_SETS[args.variant_set]
    unknown = tuple(variant_id for variant_id in variant_ids if variant_id not in VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")
    selected_cases = {part.strip() for part in args.cases.split(",") if part.strip()}
    cases = [
        case
        for case in _load_cases(args.manifest)
        if not selected_cases or case.get("id") in selected_cases
    ]
    if not cases:
        parser.error("no cases selected")

    label = args.label or f"quality-ab-matrix-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.resume_report is None:
        jsonl_path, json_path = _report_paths(args.output_root, label)
    else:
        jsonl_path = args.resume_report if args.resume_report.is_absolute() else REPO_ROOT / args.resume_report
        json_path = jsonl_path.with_suffix(".json")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[MatrixRecord] = _load_jsonl_records(jsonl_path)
    completed = _completed_keys(records)

    print(f"matrix report jsonl: {jsonl_path}")
    print(f"matrix report json: {json_path}")
    if records:
        print(f"loaded resume records: {len(records)}")
        print(f"completed rows eligible for skip: {len(completed)}")
    for case in cases:
        case_id = str(case["id"])
        prompt = str(case["prompt"])
        image = _case_reference(case)
        for variant_id in variant_ids:
            variant = VARIANTS[variant_id]
            if (case_id, variant_id) in completed:
                print(f"\n=== {case_id} :: {variant_id} ===", flush=True)
                print("skipping completed row from resume report", flush=True)
                continue
            print(f"\n=== {case_id} :: {variant_id} ===", flush=True)
            command = [
                "benchmarks/run_qwen_edit_benchmark.sh",
                "--runtime",
                "shardedit",
                "--image",
                image,
                "--prompt",
                prompt,
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--steps",
                str(args.steps),
                "--guidance",
                str(args.guidance),
                "--output-root",
                str(args.output_root),
                "--cooldown-seconds",
                str(args.cooldown_seconds),
                "--run-sequence-label",
                f"{label}:{case_id}:{variant_id}",
                "--condition-note",
                f"{variant.description}; case={case_id}",
                *variant.args,
            ]
            if args.dry_run:
                command.append("--dry-run")
            status, output = _run_command(command)
            run_dir = _parse_run_dir(output)
            record = MatrixRecord(
                case_id=case_id,
                variant_id=variant_id,
                status=status,
                run_dir=run_dir,
                output=_output_path(run_dir),
                command=tuple(command),
            )
            records.append(record)
            _write_jsonl(jsonl_path, record)
            _write_json(json_path, records)
            if status != 0:
                print(f"matrix stopped after failed run: {case_id} {variant_id}", file=sys.stderr)
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
