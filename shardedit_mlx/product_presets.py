"""Product-facing clarity/speed presets mapped onto qwen-image-shardedit-mlx runtime options."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from shardedit_mlx.mflux_fast_edit import RuntimeOptions, reference_conditioning_dimensions

ClarityTier = Literal["standard", "high"]
SpeedTier = Literal["quality", "balanced", "fast"]
LightningSteps = Literal[4, 8]

CLARITY_TIERS: tuple[ClarityTier, ...] = ("standard", "high")
SPEED_TIERS: tuple[SpeedTier, ...] = ("quality", "balanced", "fast")
LIGHTNING_STEPS_CHOICES: tuple[LightningSteps, ...] = (4, 8)

DEFAULT_CLARITY: ClarityTier = "standard"
DEFAULT_SPEED: SpeedTier = "quality"
DEFAULT_LIGHTNING_STEPS: LightningSteps = 8
DEFAULT_GUIDANCE = 1.0
DEFAULT_SEED = 42
CLARITY_BOX_SIZES: dict[ClarityTier, int] = {
    "standard": 768,
    "high": 1024,
}
DIMENSION_MULTIPLE = 32


@dataclass(frozen=True)
class ProductPlan:
    """Resolved product request: user tiers plus concrete runtime settings."""

    clarity: ClarityTier
    speed: SpeedTier
    seed: int
    width: int
    height: int
    source_width: int
    source_height: int
    clarity_box: int
    steps: int
    guidance: float
    cache_preset: str
    runtime_options: RuntimeOptions
    notes: tuple[str, ...]

    def mapping_dict(self) -> dict[str, object]:
        """Structured internal mapping for display or JSON reports."""

        options = self.runtime_options
        return {
            "clarity": self.clarity,
            "clarity_box": self.clarity_box,
            "speed": self.speed,
            "seed": self.seed,
            "source_size": f"{self.source_width}x{self.source_height}",
            "output_size": f"{self.width}x{self.height}",
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "cache_preset": self.cache_preset,
            "runtime": {
                "residency": options.residency_mode,
                "residency_window_size": options.residency_window_size,
                "cache_threshold": options.cache_threshold,
                "cache_max_consecutive": options.cache_max_consecutive,
                "cache_warmup_steps": options.cache_warmup_steps,
                "cache_back_blocks": options.cache_back_blocks,
                "cache_anchor_mode": options.cache_anchor_mode,
                "cache_predictor": options.cache_predictor,
                "cache_threshold_schedule": options.cache_threshold_schedule,
                "cache_region_policy": options.cache_region_policy,
                "reference_conditioning_size": options.reference_conditioning_size,
                "reference_conditioning_max_width": options.reference_conditioning_max_width,
                "reference_conditioning_max_height": options.reference_conditioning_max_height,
            },
            "notes": list(self.notes),
        }


def clarity_box_size(clarity: ClarityTier) -> int:
    try:
        return CLARITY_BOX_SIZES[clarity]
    except KeyError as exc:
        raise ValueError(f"unknown clarity tier: {clarity}") from exc


def fit_clarity_dimensions(
    image_width: int,
    image_height: int,
    *,
    box: int,
) -> tuple[int, int]:
    """Fit source aspect into a square box; width/height are multiples of 32."""

    fitted = reference_conditioning_dimensions(
        policy="fit-box",
        image_width=image_width,
        image_height=image_height,
        max_width=box,
        max_height=box,
    )
    if fitted is None:  # pragma: no cover - fit-box always returns a size
        raise RuntimeError("fit-box clarity sizing returned no dimensions")
    width, height = fitted
    if width % DIMENSION_MULTIPLE != 0 or height % DIMENSION_MULTIPLE != 0:
        raise RuntimeError(
            f"clarity size must be multiples of {DIMENSION_MULTIPLE}, got {width}x{height}"
        )
    return width, height


def _speed_cache(
    speed: SpeedTier,
    *,
    reference_max_width: int,
    reference_max_height: int,
) -> tuple[str, RuntimeOptions, tuple[str, ...]]:
    """Return (preset name, options, notes) for a speed tier."""

    base = RuntimeOptions(
        residency_mode="shard",
        residency_window_size=8,
        reference_conditioning_size="fit-box",
        reference_conditioning_max_width=reference_max_width,
        reference_conditioning_max_height=reference_max_height,
        profile=True,
    )
    if speed == "quality":
        return (
            "none",
            base,
            (
                "quality: full 8×60 Transformer blocks, no residual cache",
                "fit-box conditioning capped at "
                f"{reference_max_width}×{reference_max_height}",
            ),
        )
    if speed == "balanced":
        options = replace(
            base,
            cache_threshold=0.8,
            cache_max_consecutive=1,
            cache_warmup_steps=1,
            cache_back_blocks=2,
            cache_threshold_schedule="flow-aware",
        )
        return (
            "flow-aware",
            options,
            (
                "balanced: F1B2 shape with flow-aware schedule (fidelity-oriented cache)",
                "may soften fine detail vs quality; still opt-in for face identity",
            ),
        )
    if speed == "fast":
        options = replace(
            base,
            cache_threshold=0.8,
            cache_max_consecutive=1,
            cache_warmup_steps=1,
            cache_back_blocks=2,
            cache_threshold_schedule="fixed",
        )
        return (
            "f1b2",
            options,
            (
                "fast: F1B2 fixed-threshold cache (skip middle blocks on hits)",
                "fastest product tier; face identity still needs manual review",
            ),
        )
    raise ValueError(f"unknown speed tier: {speed}")


def resolve_product_plan(
    *,
    image_width: int,
    image_height: int,
    clarity: ClarityTier = DEFAULT_CLARITY,
    speed: SpeedTier = DEFAULT_SPEED,
    seed: int = DEFAULT_SEED,
    lightning_steps: LightningSteps = DEFAULT_LIGHTNING_STEPS,
) -> ProductPlan:
    """Map product tiers onto concrete sizes and RuntimeOptions."""

    if clarity not in CLARITY_TIERS:
        raise ValueError(f"clarity must be one of {CLARITY_TIERS}, got {clarity!r}")
    if speed not in SPEED_TIERS:
        raise ValueError(f"speed must be one of {SPEED_TIERS}, got {speed!r}")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if lightning_steps not in LIGHTNING_STEPS_CHOICES:
        raise ValueError(
            f"lightning_steps must be one of {LIGHTNING_STEPS_CHOICES}, got {lightning_steps!r}"
        )
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    box = clarity_box_size(clarity)
    width, height = fit_clarity_dimensions(image_width, image_height, box=box)
    cache_preset, runtime_options, speed_notes = _speed_cache(
        speed,
        reference_max_width=width,
        reference_max_height=height,
    )
    notes = (
        f"clarity {clarity}: fit source {image_width}x{image_height} into {box}x{box} "
        f"→ {width}x{height} (multiples of {DIMENSION_MULTIPLE}, aspect preserved)",
        f"lightning: {lightning_steps}-step distilled LoRA, steps={lightning_steps}",
        *speed_notes,
    )
    return ProductPlan(
        clarity=clarity,
        speed=speed,
        seed=seed,
        width=width,
        height=height,
        source_width=image_width,
        source_height=image_height,
        clarity_box=box,
        steps=int(lightning_steps),
        guidance=DEFAULT_GUIDANCE,
        cache_preset=cache_preset,
        runtime_options=runtime_options,
        notes=notes,
    )


def format_mapping_report(plan: ProductPlan) -> str:
    """Human-readable internal mapping shown after (or before) a run."""

    mapping = plan.mapping_dict()
    runtime = mapping["runtime"]
    assert isinstance(runtime, dict)
    lines = [
        "=== qwen-image-shardedit-mlx product mapping ===",
        (
            f"clarity: {plan.clarity} -> box {plan.clarity_box}x{plan.clarity_box}, "
            f"source {plan.source_width}x{plan.source_height} -> output {plan.width}x{plan.height}"
        ),
        f"speed:   {plan.speed} -> cache_preset={plan.cache_preset}",
        f"seed:    {plan.seed}",
        f"steps:   {plan.steps}  guidance: {plan.guidance}  (lightning {plan.steps}-step)",
        "runtime:",
        f"  residency: {runtime['residency']}",
        f"  cache_threshold: {runtime['cache_threshold']}",
        f"  cache_max_consecutive: {runtime['cache_max_consecutive']}",
        f"  cache_warmup_steps: {runtime['cache_warmup_steps']}",
        f"  cache_back_blocks: {runtime['cache_back_blocks']}",
        f"  cache_anchor_mode: {runtime['cache_anchor_mode']}",
        f"  cache_predictor: {runtime['cache_predictor']}",
        f"  cache_threshold_schedule: {runtime['cache_threshold_schedule']}",
        f"  reference_conditioning: {runtime['reference_conditioning_size']}"
        f" max={runtime['reference_conditioning_max_width']}x"
        f"{runtime['reference_conditioning_max_height']}",
    ]
    if plan.notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in plan.notes)
    return "\n".join(lines)
