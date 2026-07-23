"""Immutable performance models used by qwen-image-shardedit-mlx probes and runtime policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


LATENT_TOKEN_STRIDE = 16


def _validate_dimension(value: int) -> None:
    if value <= 0 or value % LATENT_TOKEN_STRIDE != 0:
        raise ValueError("image dimensions must be positive multiples of 16")


@dataclass(frozen=True)
class EditTokenPlan:
    """Token counts for one target image and all conditioning images."""

    target_tokens: int
    condition_tokens: int

    @property
    def image_tokens(self) -> int:
        return self.target_tokens + self.condition_tokens

    @property
    def attention_cost(self) -> int:
        return self.image_tokens**2

    @classmethod
    def from_dimensions(
        cls,
        *,
        target_width: int,
        target_height: int,
        condition_width: int,
        condition_height: int,
        condition_count: int = 1,
    ) -> "EditTokenPlan":
        dimensions = (target_width, target_height, condition_width, condition_height)
        for dimension in dimensions:
            _validate_dimension(dimension)
        if condition_count <= 0:
            raise ValueError("condition count must be positive")
        return cls(
            target_tokens=(target_width // LATENT_TOKEN_STRIDE) * (target_height // LATENT_TOKEN_STRIDE),
            condition_tokens=(condition_width // LATENT_TOKEN_STRIDE)
            * (condition_height // LATENT_TOKEN_STRIDE)
            * condition_count,
        )


@dataclass(frozen=True)
class RuntimeMemoryPlan:
    """Memory policy for one staged Transformer configuration."""

    physical_memory_gib: float
    system_reserve_gib: float
    activation_reserve_gib: float
    transformer_size_gib: float
    transformer_bits: int

    def __post_init__(self) -> None:
        positive_values = (
            self.physical_memory_gib,
            self.transformer_size_gib,
            self.transformer_bits,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("physical memory, Transformer size, and bit width must be positive")
        if self.system_reserve_gib < 0 or self.activation_reserve_gib < 0:
            raise ValueError("memory reserves cannot be negative")

    @property
    def required_memory_gib(self) -> float:
        return self.system_reserve_gib + self.activation_reserve_gib + self.transformer_size_gib

    @property
    def fits(self) -> bool:
        return self.required_memory_gib <= self.physical_memory_gib

    def recommended_bits(self, candidates: tuple[int, ...] = (6, 5, 4)) -> int | None:
        for bits in candidates:
            if bits <= 0:
                raise ValueError("candidate bit widths must be positive")
            scaled_transformer_gib = self.transformer_size_gib * bits / self.transformer_bits
            required_gib = self.system_reserve_gib + self.activation_reserve_gib + scaled_transformer_gib
            if required_gib <= self.physical_memory_gib:
                return bits
        return None


def classifier_free_guidance_passes(guidance: float) -> int:
    """Return the number of Transformer passes required per denoise step."""

    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be a finite non-negative number")
    return 1 if math.isclose(guidance, 1.0, rel_tol=0.0, abs_tol=1e-9) else 2
