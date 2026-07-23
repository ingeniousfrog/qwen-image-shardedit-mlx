#!/usr/bin/env python3
"""Create a qwen.image.swift-compatible overlay for a local mflux Qwen snapshot.

The overlay uses symlinks for large weight/tokenizer files and writes the
config files required by qwen.image.swift. It does not copy model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - handled by CLI at runtime
    safe_open = None  # type: ignore[assignment]


DEFAULT_MODEL = Path("models/qwen-edit-2511-q6")
DEFAULT_OUTPUT = Path("swift-overlays/qwen-edit-2511-q6")


TRANSFORMER_CONFIG: dict[str, Any] = {
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "patch_size": 2,
}


TEXT_ENCODER_CONFIG: dict[str, Any] = {
    "vocab_size": 152064,
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 18944,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 128000,
    "rms_norm_eps": 1e-6,
    "prompt_drop_index": 34,
    "torch_dtype": "bfloat16",
    "vision_config": {
        "tokens_per_second": 2,
    },
}


SCHEDULER_CONFIG: dict[str, Any] = {
    "use_dynamic_shifting": False,
    "shift": 1.0,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "patch_size": 16,
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def link_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def link_component_files(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        if source.name == "config.json":
            continue
        if source.suffix == ".safetensors" or source.name.endswith(".json"):
            link_file(source.resolve(), destination_dir / source.name)


def link_tokenizer(source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.exists() or destination_dir.is_symlink():
        if destination_dir.is_symlink() or destination_dir.is_file():
            destination_dir.unlink()
        else:
            shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True)
    for source in sorted(source_dir.iterdir()):
        if source.is_file():
            link_file(source.resolve(), destination_dir / source.name)
    write_added_tokens_json(source_dir, destination_dir)


def write_added_tokens_json(source_dir: Path, destination_dir: Path) -> None:
    existing = source_dir / "added_tokens.json"
    if existing.exists():
        link_file(existing.resolve(), destination_dir / "added_tokens.json")
        return

    tokenizer_json = source_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        return
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    added_tokens = data.get("added_tokens")
    if not isinstance(added_tokens, list):
        return
    token_map: dict[str, int] = {}
    for token in added_tokens:
        if not isinstance(token, dict):
            continue
        content = token.get("content")
        token_id = token.get("id")
        if isinstance(content, str) and isinstance(token_id, int):
            token_map[content] = token_id
    if token_map:
        write_json(destination_dir / "added_tokens.json", token_map)


def component_safetensor_keys(component_dir: Path) -> set[str]:
    if safe_open is None:
        fail("safetensors is required. Run with the Python environment that has mflux installed.")
    keys: set[str] = set()
    for file in sorted(component_dir.glob("*.safetensors")):
        with safe_open(file, framework="np") as handle:
            keys.update(handle.keys())
    return keys


def quantized_layer_names(component_dir: Path) -> list[str]:
    keys = component_safetensor_keys(component_dir)
    return sorted(key.removesuffix(".scales") for key in keys if key.endswith(".scales"))


def build_quantization_manifest(model_dir: Path, bits: int, group_size: int, mode: str) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for component in ["transformer", "text_encoder"]:
        component_dir = model_dir / component
        if not component_dir.exists():
            continue
        for name in quantized_layer_names(component_dir):
            layers.append(
                {
                    "component": component,
                    "name": name,
                    "group_size": group_size,
                    "bits": bits,
                    "mode": mode,
                }
            )
    return {
        "version": 1,
        "snapshot": str(model_dir),
        "group_size": group_size,
        "bits": bits,
        "mode": mode,
        "layers": layers,
    }


def validate_source_layout(model_dir: Path) -> None:
    required = [
        model_dir / "tokenizer" / "tokenizer.json",
        model_dir / "tokenizer" / "tokenizer_config.json",
        model_dir / "transformer" / "model.safetensors.index.json",
        model_dir / "text_encoder" / "model.safetensors.index.json",
        model_dir / "vae" / "model.safetensors.index.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        fail("source model is missing required files:\n" + "\n".join(f"  {path}" for path in missing))


def prepare_overlay(
    model_dir: Path,
    output_dir: Path,
    bits: int,
    group_size: int,
    mode: str,
    force: bool,
) -> None:
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    validate_source_layout(model_dir)

    if output_dir.exists():
        if not force:
            fail(f"overlay already exists: {output_dir} (pass --force to replace)")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    link_tokenizer(model_dir / "tokenizer", output_dir / "tokenizer")
    for component in ["transformer", "text_encoder", "vae"]:
        link_component_files(model_dir / component, output_dir / component)

    write_json(output_dir / "transformer" / "config.json", TRANSFORMER_CONFIG)
    write_json(output_dir / "text_encoder" / "config.json", TEXT_ENCODER_CONFIG)
    write_json(output_dir / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    write_json(output_dir / "quantization.json", build_quantization_manifest(model_dir, bits, group_size, mode))
    write_json(
        output_dir / "shardedit-overlay.json",
        {
            "source_model": str(model_dir),
            "notes": [
                "Generated by qwen-image-shardedit-mlx for qwen.image.swift compatibility.",
                "Large files are symlinks; do not move the source model while using this overlay.",
                "The local README identifies the source q6 model as Qwen-Image-Edit-2509.",
            ],
        },
    )

    print(f"overlay: {output_dir}")
    print(f"source:  {model_dir}")
    print("next:")
    print(f"  python3 tools/audit_model_layout.py {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Source mflux Qwen snapshot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Overlay output directory")
    parser.add_argument("--bits", type=int, default=6, help="Prepacked quantization bit width")
    parser.add_argument("--group-size", type=int, default=64, help="Prepacked quantization group size")
    parser.add_argument("--mode", choices=["affine", "mxfp4"], default="affine", help="Quantization mode")
    parser.add_argument("--force", action="store_true", help="Replace an existing overlay")
    args = parser.parse_args()

    prepare_overlay(
        model_dir=args.model,
        output_dir=args.output,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
