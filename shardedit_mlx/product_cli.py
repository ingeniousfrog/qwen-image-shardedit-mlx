"""Product CLI: image/prompt/clarity/speed/seed with mapped runtime + metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from shardedit_mlx.product_presets import (
    CLARITY_TIERS,
    DEFAULT_CLARITY,
    DEFAULT_LIGHTNING_STEPS,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    LIGHTNING_STEPS_CHOICES,
    SPEED_TIERS,
    ProductPlan,
    format_mapping_report,
    resolve_product_plan,
)
from shardedit_mlx.time_metrics import TimeLMetrics, format_metrics_report, parse_time_l


def read_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) for a local image without decoding pixels."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow comes with mflux runtime
        raise SystemExit("Pillow is required to read image dimensions") from exc
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise SystemExit(f"image dimensions must be positive: {path}")
    return width, height


def parse_image_option(value: str) -> tuple[Path, ...]:
    """Parse `--image` as one path or comma-separated paths."""

    parts = [part.strip() for part in value.split(",")]
    paths = tuple(Path(part).expanduser() for part in parts if part)
    if not paths:
        raise argparse.ArgumentTypeError("--image must contain at least one path")
    return paths


DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "SHARDEDIT_MODEL_PATH",
        "models/qwen-edit-2511-q6",
    )
)
LIGHTNING_LORA_PATHS: dict[int, Path] = {
    8: Path(
        os.environ.get(
            "SHARDEDIT_LORA_8STEP_PATH",
            "loras/"
            "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        )
    ),
    4: Path(
        os.environ.get(
            "SHARDEDIT_LORA_4STEP_PATH",
            "loras/"
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        )
    ),
}
DEFAULT_LORA_PATH = LIGHTNING_LORA_PATHS[8]
if "SHARDEDIT_LORA_PATH" in os.environ:
    DEFAULT_LORA_PATH = Path(os.environ["SHARDEDIT_LORA_PATH"])

DEFAULT_DRY_RUN_IMAGE_SIZE = (576, 768)


@dataclass(frozen=True)
class ProductRunResult:
    plan: ProductPlan
    output: Path
    command: tuple[str, ...]
    returncode: int
    metrics: TimeLMetrics
    metrics_file: str = ""

    def summary_dict(self) -> dict[str, object]:
        return {
            "ok": self.returncode == 0,
            "returncode": self.returncode,
            "output": str(self.output),
            "mapping": self.plan.mapping_dict(),
            "metrics": self.metrics.as_dict(),
            "command": list(self.command),
        }


def build_edit_argv(
    plan: ProductPlan,
    *,
    images: tuple[Path, ...] | list[Path],
    prompt: str,
    model: Path,
    lora: Path,
    output: Path,
    lora_scale: float = 1.0,
    low_ram: bool = True,
) -> list[str]:
    """Build argv for `python -m shardedit_mlx.mflux_fast_edit` (no executable)."""

    image_paths = tuple(Path(path) for path in images)
    if not image_paths:
        raise ValueError("at least one image path is required")
    options = plan.runtime_options
    argv = [
        "--model",
        str(model),
        "--base-model",
        "qwen",
        "--image-paths",
        *[str(path) for path in image_paths],
        "--prompt",
        prompt,
        "--seed",
        str(plan.seed),
        "--steps",
        str(plan.steps),
        "--guidance",
        str(plan.guidance),
        "--width",
        str(plan.width),
        "--height",
        str(plan.height),
        "--lora-paths",
        str(lora),
        "--lora-scales",
        str(lora_scale),
        "--output",
        str(output),
        "--shardedit-residency",
        options.residency_mode,
        "--shardedit-residency-window-size",
        str(options.residency_window_size),
        "--shardedit-cache-threshold",
        str(options.cache_threshold),
        "--shardedit-cache-max-consecutive",
        str(options.cache_max_consecutive),
        "--shardedit-cache-warmup-steps",
        str(options.cache_warmup_steps),
        "--shardedit-cache-back-blocks",
        str(options.cache_back_blocks),
        "--shardedit-cache-anchor-mode",
        options.cache_anchor_mode,
        "--shardedit-cache-predictor",
        options.cache_predictor,
        "--shardedit-cache-threshold-schedule",
        options.cache_threshold_schedule,
        "--shardedit-cache-region-policy",
        options.cache_region_policy,
        "--shardedit-reference-conditioning-size",
        options.reference_conditioning_size,
        "--shardedit-reference-conditioning-max-width",
        str(options.reference_conditioning_max_width),
        "--shardedit-reference-conditioning-max-height",
        str(options.reference_conditioning_max_height),
    ]
    if low_ram:
        argv.append("--low-ram")
    if options.profile:
        argv.append("--shardedit-profile")
    return argv


def build_timed_command(
    edit_argv: list[str],
    *,
    metrics_path: Path,
    python: str | None = None,
) -> list[str]:
    """Wrap the edit module with macOS `/usr/bin/time -l -o <file>`."""

    interpreter = python or sys.executable
    return [
        "/usr/bin/time",
        "-l",
        "-o",
        str(metrics_path),
        interpreter,
        "-m",
        "shardedit_mlx.mflux_fast_edit",
        *edit_argv,
    ]


def run_product_edit(
    plan: ProductPlan,
    *,
    images: tuple[Path, ...] | list[Path],
    prompt: str,
    model: Path,
    lora: Path,
    output: Path,
    lora_scale: float = 1.0,
    low_ram: bool = True,
    cwd: Path | None = None,
) -> ProductRunResult:
    """Execute one product edit; stream progress to the terminal and parse time -l."""

    resolved_images = tuple(path.expanduser().resolve() for path in images)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    edit_argv = build_edit_argv(
        plan,
        images=resolved_images,
        prompt=prompt,
        model=model.expanduser().resolve(),
        lora=lora.expanduser().resolve(),
        output=output,
        lora_scale=lora_scale,
        low_ram=low_ram,
    )
    metrics_file = tempfile.NamedTemporaryFile(
        prefix="shardedit-time-",
        suffix=".txt",
        delete=False,
    )
    metrics_path = Path(metrics_file.name)
    metrics_file.close()
    command = build_timed_command(edit_argv, metrics_path=metrics_path)
    try:
        # Inherit stdio so tqdm / denoise progress stays visible on the terminal.
        # `/usr/bin/time -l -o` writes resource counters to metrics_path only.
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
        metrics_text = (
            metrics_path.read_text(encoding="utf-8", errors="replace")
            if metrics_path.exists()
            else ""
        )
        metrics = parse_time_l(metrics_text)
    finally:
        metrics_path.unlink(missing_ok=True)
    return ProductRunResult(
        plan=plan,
        output=output,
        command=tuple(command),
        returncode=completed.returncode,
        metrics=metrics,
        metrics_file=str(metrics_path),
    )


def format_run_report(result: ProductRunResult) -> str:
    sections = [
        format_mapping_report(result.plan),
        "",
        format_metrics_report(result.metrics),
        "",
        f"output: {result.output}",
        f"exit:   {result.returncode}",
    ]
    return "\n".join(sections)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardedit-edit",
        description=(
            "Product-facing Qwen Image Edit entry: image, prompt, clarity, speed, seed. "
            "Internal runtime mapping and /usr/bin/time -l metrics are printed after the run."
        ),
    )
    parser.add_argument(
        "--image",
        type=parse_image_option,
        default=(Path("ref.png"),),
        help=(
            "Reference image path, or comma-separated paths for multi-image edit "
            "(default: ./ref.png). The first image sets output aspect ratio / clarity sizing."
        ),
    )
    parser.add_argument("--prompt", required=True, help="Edit instruction")
    parser.add_argument(
        "--clarity",
        choices=CLARITY_TIERS,
        default=DEFAULT_CLARITY,
        help=(
            "Output size budget: standard fits into 768x768, high into 1024x1024; "
            "aspect follows the source image; width/height are multiples of 32"
        ),
    )
    parser.add_argument(
        "--speed",
        choices=SPEED_TIERS,
        default=DEFAULT_SPEED,
        help="quality=no cache, balanced=flow-aware, fast=F1B2",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lightning-steps",
        type=int,
        choices=LIGHTNING_STEPS_CHOICES,
        default=DEFAULT_LIGHTNING_STEPS,
        help=(
            "Lightning distillation variant: 8 uses the 8-step LoRA (default), "
            "4 uses the 4-step LoRA and sets steps=4. Ignored if --lora is passed."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: ./shardedit-edit.png)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen Image Edit model path (or SHARDEDIT_MODEL_PATH)",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help=(
            "Explicit LoRA path. Overrides --lightning-steps. "
            "Defaults to the Lightning LoRA that matches --lightning-steps."
        ),
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-low-ram",
        action="store_true",
        help="Do not pass --low-ram to the underlying mflux path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mapping and command only; do not run inference",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary to stdout (mapping + metrics when run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output or Path("shardedit-edit.png")
    images: tuple[Path, ...] = args.image
    primary_image = images[0]

    missing = [str(path) for path in images if not path.exists()]
    if missing and not args.dry_run:
        print(
            "error: image path does not exist: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    if missing:
        print(
            "dry-run warning: image path does not exist; "
            f"using {DEFAULT_DRY_RUN_IMAGE_SIZE[0]}x{DEFAULT_DRY_RUN_IMAGE_SIZE[1]} "
            "for command mapping: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    source_width, source_height = (
        read_image_size(primary_image)
        if primary_image.exists()
        else DEFAULT_DRY_RUN_IMAGE_SIZE
    )
    lora_path = args.lora or LIGHTNING_LORA_PATHS[args.lightning_steps]
    plan = resolve_product_plan(
        image_width=source_width,
        image_height=source_height,
        clarity=args.clarity,
        speed=args.speed,
        seed=args.seed,
        lightning_steps=args.lightning_steps,
    )

    if args.dry_run:
        metrics_placeholder = Path("/tmp/shardedit-time-metrics.txt")
        edit_argv = build_edit_argv(
            plan,
            images=images,
            prompt=args.prompt,
            model=args.model,
            lora=lora_path,
            output=output,
            lora_scale=args.lora_scale,
            low_ram=not args.no_low_ram,
        )
        command = build_timed_command(edit_argv, metrics_path=metrics_placeholder)
        payload = {
            "ok": True,
            "dry_run": True,
            "images": [str(path) for path in images],
            "primary_image": str(primary_image),
            "lora": str(lora_path),
            "lightning_steps": args.lightning_steps,
            "mapping": plan.mapping_dict(),
            "command": command,
            "output": str(output),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(format_mapping_report(plan))
            print(f"images: {', '.join(str(path) for path in images)}")
            print(f"primary_image (clarity/aspect): {primary_image}")
            print(f"lora: {lora_path}")
            print()
            print("command:")
            print("  " + " ".join(command))
        return 0

    for path, label in (
        (args.model, "model"),
        (lora_path, "lora"),
    ):
        if not path.expanduser().exists():
            print(f"error: {label} path does not exist: {path}", file=sys.stderr)
            return 2

    if not args.json:
        print(format_mapping_report(plan), flush=True)
        print(f"images: {', '.join(str(path) for path in images)}", flush=True)
        print(f"primary_image (clarity/aspect): {primary_image}", flush=True)
        print(f"lora: {lora_path}", flush=True)
        print("\nrunning...\n", flush=True)

    result = run_product_edit(
        plan,
        images=images,
        prompt=args.prompt,
        model=args.model,
        lora=lora_path,
        output=output,
        lora_scale=args.lora_scale,
        low_ram=not args.no_low_ram,
    )

    if args.json:
        payload = result.summary_dict()
        payload["images"] = [str(path) for path in images]
        payload["primary_image"] = str(primary_image)
        payload["lora"] = str(lora_path)
        payload["lightning_steps"] = args.lightning_steps
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_run_report(result), flush=True)

    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
