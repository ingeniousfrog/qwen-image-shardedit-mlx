# qwen-image-shardedit-mlx

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文 README](README-CN.md)

qwen-image-shardedit-mlx is an unofficial Apple Silicon / MLX runtime for
[Qwen Image Edit](https://github.com/QwenLM/Qwen-Image). It keeps an
mflux-compatible path, adds shard residency for 24 GB Macs, and exposes residual
cache modes as opt-in speed presets. The package installs a CLI named
`shardedit-edit`.

This repository ships code and docs only. Model weights, LoRAs, reference
photos, and generated images are not included.

## Features

- Run Qwen Image Edit locally on Apple Silicon through an mflux-compatible layer.
- Stream Transformer blocks with shard residency instead of keeping the full stack hot.
- Map `clarity` / `speed` / `seed` to concrete runtime flags through `shardedit-edit`.
- Keep a conservative default: shard residency with no residual cache.
- Opt into `balanced` or `fast` residual-cache presets when you want more speed.
- Cap reference token budget with fit-box conditioning (`576x768` stays `576x768`).
- Dry-run command mapping before spending wall time on inference.
- Keep rejected probes in-tree for reproducibility, not as recommended defaults.

## Measured Results

Hardware: base Apple M2 with 24 GB unified memory. Workload: portrait reference,
`576x768` fit-box, Lightning 8-step LoRA. Treat these as a reproducible starting
point, not a promise for other machines.

| Path | Time | Role |
| --- | ---: | --- |
| stock mflux `--low-ram` | ~32m 37s wall | baseline |
| shard, no cache | ~6m 08s process | default fidelity |
| shard + F1B2 / max=1 | ~3m 53s process | opt-in speed |
| shard + flow-aware cache | ~4m 18s process | opt-in fidelity-oriented cache |

Sub-minute latency is not claimed. Cache presets can pass a coarse pixel screen;
face identity, hair, and clothing still need manual review before you make them
default.

## Install

Create a virtualenv and install the runtime extras:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

Optional K-quant diagnostics only:

```bash
python -m pip install -e ".[experiments]"
```

Runtime Python packages are `mflux>=0.18`, `mlx>=0.31`, `safetensors>=0.8`, and
`Pillow>=10`. `mflux` is required: this project is a narrow mflux-compatible
layer, not a from-scratch Qwen Image Edit implementation.

`mlx-kquant` and full Xcode / Metal command-line tools are optional. They are
only needed to reproduce rejected kernel or K-quant probes. See
[docs/installation.md](docs/installation.md) for the dependency matrix.

Expected local layout (gitignored):

```text
models/qwen-edit-2511-q6/
loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
ref.png
```

Override paths with CLI flags or env vars
(`SHARDEDIT_MODEL_PATH`, `SHARDEDIT_LORA_PATH`, `SHARDEDIT_IMAGE_PATH`).
[.env.example](.env.example) shows the common knobs.

## Model Assets

Measured runs used:

| Asset | Source | File / folder |
| --- | --- | --- |
| Base weights (q6) | [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux) | `Qwen-Image-Edit-2511-q6` |
| Lightning LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) ([ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)) | `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` |

q6 is a middle ground. Local q4 face edits lost identity too easily; q8 was
skipped for download size and expected 24 GB pressure. The tested mflux-ready
q6 package came from Hugging Face. Do not assume ModelScope hosts the same
folder — visible alternatives there may be 2509 or other non-equivalent
variants.

Download helpers:

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux

# LoRA from ModelScope:
modelscope download \
  --model lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local_dir loras

# Or Hugging Face:
huggingface-cli download \
  lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local-dir loras
```

Rename or symlink the q6 folder to `models/qwen-edit-2511-q6`, or point
`SHARDEDIT_MODEL_PATH` at the downloaded path.

## CLI

`shardedit-edit` is not a stock mflux command. It is the console script
registered by this package in `pyproject.toml`:

```toml
[project.scripts]
shardedit-edit = "shardedit_mlx.product_cli:main"
```

It is the product-facing entry point: it accepts `image` / `prompt` / `speed` /
`seed`, then expands them into lower-level `shardedit_mlx.mflux_fast_edit` and
mflux runtime arguments.

Check command mapping without running inference:

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

Run the fidelity path (recommended first real edit):

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

Try an opt-in cache preset after you review identity:

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed balanced \
  --output outputs/ref-cafe-balanced.png
```

Dry-check the benchmark harness:

```bash
benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --dry-run
```

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--image` | `ref.png` | Reference image, or comma-separated paths |
| `--prompt` | required | Edit instruction |
| `--clarity standard` | `standard` | Fit into a `768x768` box; keep aspect ratio |
| `--clarity high` | off | Fit into a `1024x1024` box |
| `--speed quality` | `quality` | Shard residency, no residual cache |
| `--speed balanced` | off | Flow-aware F1B2 cache |
| `--speed fast` | off | Fixed F1B2 cache |
| `--seed` | `42` | Reproducible seed |
| `--dry-run` | off | Print mapping only |

Lower-level `--shardedit-*` flags and rejected probes (token merge V0, dense /
K-quant image MLP, custom q6 Metal kernels) stay in the tree for auditability.
They are not default speed paths. Full map:
[docs/parameters.md](docs/parameters.md).

## Speed Presets

| Preset | Runtime shape | Status |
| --- | --- | --- |
| `quality` | `shard`, no cache, fit-box conditioning | default fidelity path |
| `balanced` | `shard`, F1B2/max=1, flow-aware threshold | opt-in cache candidate |
| `fast` | `shard`, F1B2/max=1, fixed threshold | opt-in speed candidate |

Defaults stay conservative on purpose:

- `guidance=1.0` pruning is mathematically exact.
- Shard residency still executes all 8 steps × 60 Transformer blocks.
- Fit-box conditioning caps large references to a controlled token budget.
- Residual cache is approximate, so it stays opt-in until face review passes.

Experiment history and reject/keep decisions live in
[docs/experiment-rationale.md](docs/experiment-rationale.md).

## Development

```bash
python -m pip install -e ".[runtime,test]"
python -m pytest
```

More detail:

| Doc | Contents |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Dependency layers, Metal notes, first run |
| [docs/parameters.md](docs/parameters.md) | Product CLI and experiment flags |
| [docs/experiment-rationale.md](docs/experiment-rationale.md) | What was tried, kept, and rejected |
| [docs/open-source-checklist.md](docs/open-source-checklist.md) | Pre-publish hygiene |

## License

Project code is MIT. Model weights, LoRAs, mflux, MLX, and Qwen assets keep their
own licenses and terms. Do not commit private images, weights, LoRAs, or
generated outputs.
