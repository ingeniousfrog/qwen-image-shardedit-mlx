# qwen-image-shardedit-mlx

[中文说明](README-CN.md)

Unofficial Apple Silicon / MLX runtime for [Qwen Image Edit](https://github.com/QwenLM/Qwen-Image), built as a thin mflux-compatible layer. It adds **shard residency** so 24 GB Macs can run the full 8-step × 60-block edit path without the stock `--low-ram` wall-clock cliff, and exposes residual-cache presets as **opt-in** speed experiments.

> This repository ships code and docs only — no model weights, LoRAs, reference photos, or generated images.

## Highlights

- **Shard residency** — stream Transformer blocks through unified memory instead of keeping the full stack resident
- **Product CLI** — `shardedit-edit` maps clarity / speed / seed to concrete runtime flags
- **Conservative default** — `quality` = shard + no residual cache (exact block execution at `guidance=1.0`)
- **Opt-in cache presets** — `balanced` / `fast` trade fidelity for wall time; keep off until you review faces manually
- **Reproducible evidence** — benchmark harness, reject/keep rationale, and experiment flags retained for audit

## Measured Results

Hardware: base Apple **M2**, **24 GB** unified memory. Portrait reference → `576×768` fit-box. Treat these as a reproducible baseline, not a cross-machine guarantee.

| Path | Time (process) | Role |
| --- | ---: | --- |
| stock mflux `--low-ram` | ~32m 37s (wall) | baseline |
| shard, no cache | ~6m 08s | **default fidelity** |
| shard + F1B2 / max=1 | ~3m 53s | opt-in speed |
| shard + flow-aware cache | ~4m 18s | opt-in fidelity-oriented cache |

Sub-minute latency is **not** claimed. Cache presets pass a coarse pixel screen; face identity, hair, and clothing still need human review before you make them default.

## Requirements

| Item | Notes |
| --- | --- |
| Hardware | macOS on Apple Silicon with working Metal / MLX |
| Python | 3.11+ |
| Runtime deps | `mflux≥0.18`, `mlx≥0.31`, `safetensors≥0.8`, `Pillow≥10` |
| Local assets | Qwen Image Edit **q6** weights, Lightning LoRA, private `ref.png` |

`mlx-kquant` and full Xcode / Metal CLT are optional — only for reproducing rejected kernel / K-quant probes. See [docs/installation.md](docs/installation.md).

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

Optional K-quant diagnostics:

```bash
python -m pip install -e ".[experiments]"
```

Expected local layout (all gitignored):

```text
models/qwen-edit-2511-q6/
loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
ref.png
```

Override paths via flags or env (`SHARDEDIT_MODEL_PATH`, `SHARDEDIT_LORA_PATH`, `SHARDEDIT_IMAGE_PATH`). Copy [.env.example](.env.example) if useful.

## Model Assets

Measured runs used:

| Asset | Source | File / folder |
| --- | --- | --- |
| Base weights (q6) | [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux) | `Qwen-Image-Edit-2511-q6` |
| Lightning LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) ([ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)) | `…-8steps-V1.0-bf16.safetensors` |

**Why q6:** local q4 face edits lost identity too easily; q8 was skipped for size and expected 24 GB pressure. The tested mflux-ready q6 package came from Hugging Face — do not assume ModelScope hosts the same folder (visible alternatives there may be 2509 or other non-equivalent variants).

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux
# then rename/symlink to models/qwen-edit-2511-q6, or set SHARDEDIT_MODEL_PATH

# LoRA — ModelScope or Hugging Face:
modelscope download \
  --model lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local_dir loras
```

## Quick Start

Dry-run (no inference):

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

Fidelity path (recommended first run):

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

Benchmark harness dry check:

```bash
benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --dry-run
```

### CLI presets

| Option | Meaning |
| --- | --- |
| `--clarity standard` | Fit into a `768×768` box; a `576×768` ref stays `576×768` |
| `--speed quality` | Shard residency, **no** residual cache (default) |
| `--speed balanced` | Flow-aware F1B2 cache (opt-in) |
| `--speed fast` | Fixed F1B2 cache (opt-in) |
| `--seed` | Reproducible seed (default `42`) |

Lower-level `--shardedit-*` flags and rejected probes (token merge V0, dense / K-quant image MLP, custom q6 Metal kernels) stay in-tree for reproducibility — they are not recommended defaults. Full map: [docs/parameters.md](docs/parameters.md).

## Design Notes

Defaults stay conservative on purpose:

1. `guidance=1.0` pruning is mathematically exact.
2. Shard residency still runs all 8 steps × 60 Transformer blocks.
3. Fit-box conditioning caps reference token budget.
4. Residual cache is approximate → opt-in until face review passes.

Experiment history and reject/keep decisions: [docs/experiment-rationale.md](docs/experiment-rationale.md).

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Dependency matrix, Metal notes, first run |
| [docs/parameters.md](docs/parameters.md) | Product CLI + experiment flags |
| [docs/experiment-rationale.md](docs/experiment-rationale.md) | What was tried, kept, and rejected |
| [docs/open-source-checklist.md](docs/open-source-checklist.md) | Pre-publish hygiene |

## License

Project code is **MIT**. Model weights, LoRAs, mflux, MLX, and Qwen assets retain their own licenses and terms. Do not commit private images, weights, LoRAs, or generated outputs.
