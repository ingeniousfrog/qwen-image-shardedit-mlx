#!/usr/bin/env python3
"""Audit a local Qwen Image model snapshot for runtime compatibility."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = Path("models/qwen-edit-2511-q6")


@dataclass(frozen=True)
class Requirement:
    runtime: str
    path: str
    required: bool = True


REQUIREMENTS = [
    Requirement("mflux", "configuration.json", required=False),
    Requirement("mflux", "tokenizer/tokenizer.json"),
    Requirement("mflux", "tokenizer/tokenizer_config.json"),
    Requirement("mflux", "transformer/model.safetensors.index.json"),
    Requirement("mflux", "text_encoder/model.safetensors.index.json"),
    Requirement("mflux", "vae/model.safetensors.index.json"),
    Requirement("qwen.image.swift", "transformer/config.json"),
    Requirement("qwen.image.swift", "text_encoder/config.json"),
    Requirement("qwen.image.swift", "scheduler/scheduler_config.json"),
    Requirement("qwen.image.swift", "tokenizer/tokenizer.json"),
    Requirement("qwen.image.swift", "vae/config.json", required=False),
]


def format_bytes(size: int) -> str:
    amount = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{size} B"


def read_base_model(model_dir: Path) -> str:
    readme = model_dir / "README.md"
    if not readme.exists():
        return "-"
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("base_model:"):
            return line.split(":", 1)[1].strip()
    return "-"


def print_requirements(model_dir: Path) -> bool:
    print("Runtime compatibility:")
    print("runtime           status  path")
    print("----------------  ------  ----")
    all_required_present = True
    for requirement in REQUIREMENTS:
        exists = (model_dir / requirement.path).exists()
        status = "ok" if exists else ("missing" if requirement.required else "optional-missing")
        if requirement.required and not exists:
            all_required_present = False
        print(f"{requirement.runtime.ljust(16)}  {status.ljust(14)}  {requirement.path}")
    return all_required_present


def print_component_sizes(model_dir: Path) -> None:
    print()
    print("Component safetensors:")
    print("component     count  size")
    print("------------  -----  ----")
    for name in ["transformer", "text_encoder", "vae"]:
        files = sorted((model_dir / name).glob("*.safetensors"))
        total = sum(path.stat().st_size for path in files)
        print(f"{name.ljust(12)}  {str(len(files)).rjust(5)}  {format_bytes(total)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when a required file is missing")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        parser.error(f"model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        parser.error(f"model path is not a directory: {model_dir}")

    print(f"Model: {model_dir}")
    print(f"base_model: {read_base_model(model_dir)}")
    print()
    all_required_present = print_requirements(model_dir)
    print_component_sizes(model_dir)

    if args.strict and not all_required_present:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
