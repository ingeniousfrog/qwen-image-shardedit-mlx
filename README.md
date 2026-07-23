# qwen-image-shardedit-mlx

**Last Updated:** 2026-07-23

[中文说明](README-CN.md)

qwen-image-shardedit-mlx is an unofficial Apple Silicon / MLX runtime experiment for
Qwen Image Edit. It keeps the mflux-compatible path, adds shard residency for
24 GB Macs, and exposes residual-cache experiments as opt-in presets.

This repository does not include model weights, LoRA files, reference photos,
or generated benchmark images. Put your private test image at `ref.png`; the
default benchmark size is `576x768`.

## Current Status

The measured local evidence comes from one base Apple M2 machine with 24 GB
unified memory. Treat the numbers as a reproducible starting point, not a
promise for other hardware.

| Path | Output | Time on measured M2 24 GB | Decision |
| --- | --- | ---: | --- |
| original mflux `--low-ram` | 576x768 | 1957.33s wall, about 32m37s | baseline |
| shard, no cache | 576x768 fit-box | 367.94s process, about 6m08s | fidelity default |
| shard + F1B2/max=1 | 576x768 fit-box | 232.66s process, about 3m53s | opt-in speed |
| shard + flow-aware | 576x768 fit-box | 258.24s process, about 4m18s | opt-in fidelity-oriented cache |

One minute has not been reached. Cache modes pass the current coarse pixel
screen, but face identity still needs manual review before any cache preset
should become your default.

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

Runtime dependencies are `mflux>=0.18`, `mlx>=0.31`, `safetensors>=0.8`, and
`Pillow>=10`. `mlx-kquant` is optional and only needed for K-quant diagnostic
experiments:

```bash
python -m pip install -e ".[experiments]"
```

Full Xcode / Metal command-line tools are not required for the default Python
runtime, but they are needed if you reproduce the rejected Swift/metallib path
or do custom Metal kernel work. See
[docs/installation.md](docs/installation.md) for the dependency matrix.

Place local assets using these names:

```text
models/qwen-edit-2511-q6/
loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
ref.png
```

`ref.png` should be your own private reference image. The default smoke path
assumes a portrait-like `576x768` reference.

## Model Assets

The measured local runs used:

- base model weights: [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux), specifically the `Qwen-Image-Edit-2511-q6` folder;
- LoRA: [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) or the
  [ModelScope mirror](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning), specifically
  `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors`.

The q6 weights were chosen as a middle ground. In local face-editing checks,
q4 was too weak for identity preservation; q8 was larger, skipped for download
size, and expected to increase memory/runtime pressure on the 24 GB M2 target.
The exact mflux-ready q6 package used here was downloaded from Hugging Face;
ModelScope should not be treated as the source for this specific base-weight
asset. At setup time, other ModelScope resources were visible, including
2509-related or non-equivalent variants, but not this tested 2511 q6 mflux
folder.

Example download helpers:

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

# Download the q6 mflux-ready model folder, then either rename it to
# models/qwen-edit-2511-q6 or point SHARDEDIT_MODEL_PATH at the downloaded folder.
huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux

# Choose one LoRA source. ModelScope:
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

## Quick Start

Check command mapping without running inference:

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

Run the product-facing fidelity path:

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

Run the benchmark harness dry check:

```bash
benchmarks/run_qwen_edit_benchmark.sh \
  --runtime shardedit \
  --dry-run
```

## Normal vs Experimental Controls

Use `shardedit-edit` for normal runs:

| User option | Meaning |
| --- | --- |
| `--image` | Reference image, default `ref.png` |
| `--prompt` | Edit instruction |
| `--clarity standard` | Fit output into a `768x768` box; a `576x768` reference stays `576x768` |
| `--speed quality` | Shard residency, no residual cache |
| `--speed balanced` | Flow-aware F1B2 cache, opt-in |
| `--speed fast` | Fixed F1B2 cache, opt-in |
| `--seed` | Reproducible seed, default `42` |

Use `benchmarks/run_qwen_edit_benchmark.sh` and the lower-level
`--shardedit-*` flags only when reproducing experiments. Rejected probes such
as token merge V0, dense image MLP, K-quant image MLP, and custom q6 Metal
kernels remain in the code for auditability, but they are not default speed
paths.

See [docs/parameters.md](docs/parameters.md) for the full parameter map.

## Why This Shape

The current default is conservative:

- `guidance=1.0` pruning is mathematically exact.
- `shard` residency still executes all 8 steps x 60 Transformer blocks.
- `fit-box` conditioning caps large references to a controlled token budget.
- F1B2 and flow-aware cache modes are faster but approximate, so they stay
  opt-in until manual face review passes.

The experiment history and reject/keep decisions are summarized in
[docs/experiment-rationale.md](docs/experiment-rationale.md). A cleaned Chinese
article draft lives at [docs/wechat-article.md](docs/wechat-article.md).

## Open-Source Hygiene

Do not commit:

- `ref.png` or other private reference images.
- model weights under `models/`.
- LoRA files under `loras/`.
- generated outputs under `benchmark-runs/` or `outputs/`.

Before publishing, run:

```bash
rg -n "lxy-1|/Users/|/private/tmp|benchmark-runs/2026|\\.safetensors" .
git status --short
```

The code in this folder is MIT licensed. Model weights, LoRA files, mflux, MLX,
and Qwen assets each have their own licenses and terms.
