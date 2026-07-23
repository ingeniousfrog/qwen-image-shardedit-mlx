from __future__ import annotations

from pathlib import Path

import pytest

from shardedit_mlx.benchmark_lora import normalize_lora_args


def test_normalize_lora_args_defaults_scale_to_one() -> None:
    paths, scales = normalize_lora_args(["~/loras/a.safetensors"], [])

    assert paths == (str(Path("~/loras/a.safetensors").expanduser()),)
    assert scales == (1.0,)


def test_normalize_lora_args_rejects_scale_without_path() -> None:
    with pytest.raises(ValueError, match="without LoRA paths"):
        normalize_lora_args([], [1.0])


def test_normalize_lora_args_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalize_lora_args(["a.safetensors", "b.safetensors"], [1.0])
