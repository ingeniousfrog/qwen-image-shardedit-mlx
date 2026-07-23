# Installation And First Run

**Last Updated:** 2026-07-23

qwen-image-shardedit-mlx is intended for Apple Silicon Macs with Metal/MLX support. The
current measured path was tested on a base Apple M2 with 24 GB unified memory.
Other machines should rerun the dry checks and benchmark harness before trusting
the presets.

## Requirements

- macOS on Apple Silicon with a working Metal/MLX environment.
- Python 3.11 or newer.
- Runtime Python packages: `mflux>=0.18`, `mlx>=0.31`, `safetensors>=0.8`, `Pillow>=10`.
- Local Qwen Image Edit q6 model files.
- Local Lightning LoRA file.
- A private reference image named `ref.png`.

The repository does not download or redistribute model weights, LoRAs, or test
photos.

## Dependency Layers

| Layer | Install | Needed for |
| --- | --- | --- |
| Base package | `python -m pip install -e .` | Importing pure helpers and reading docs |
| Runtime | `python -m pip install -e ".[runtime]"` | `shardedit-edit`, `shardedit-mflux-edit`, real image generation |
| Tests | `python -m pip install -e ".[runtime,test]"` | Unit tests that do not need optional experiment backends |
| Experiments | `python -m pip install -e ".[runtime,test,experiments]"` | `mlx-kquant` probes such as `--kquant-img-ff-window` |

`mflux` is required for the current runtime path. qwen-image-shardedit-mlx is a narrow
mflux-compatible layer, not a standalone Qwen Image Edit implementation.

`mlx-kquant` is optional. Install it only if you want to reproduce the K-quant
image-MLP diagnostics. It is not needed for `--speed quality`, `balanced`,
`fast`, or the normal quality regression scripts.

## Apple Toolchain Notes

For normal Python + MLX inference, the important check is that MLX can see a
Metal device from the terminal where you run qwen-image-shardedit-mlx. If tests fail with
`No Metal device available`, run the Metal-dependent tests from a normal macOS
terminal instead of a headless or sandboxed session.

Xcode Command Line Tools are useful for Git, build fallbacks, and local
debugging:

```bash
xcode-select --install
```

Full Xcode / Metal command-line tools are only needed for the rejected Swift
compatibility path or custom Metal kernel work. Those tools are used by
`tools/build_mlx_metallib.py`, which looks for `metal` and `metallib` through
`xcrun`.

Useful checks:

```bash
xcrun -find metal
xcrun -find metallib
```

If Xcode was just installed, open it once or complete first-launch setup before
building Swift/Metal artifacts. This is not required for the default
`shardedit-edit` path.

## Install The Package

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

Optional experiment backend:

```bash
python -m pip install -e ".[experiments]"
```

If you keep assets somewhere else, either pass `--model`, `--lora`, and
`--image`, or export:

```bash
export SHARDEDIT_MODEL_PATH=/path/to/qwen-edit-2511-q6
export SHARDEDIT_LORA_PATH=/path/to/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
export SHARDEDIT_IMAGE_PATH=/path/to/ref.png
```

## Default Asset Layout

The simplest local layout is:

```text
qwen-image-shardedit-mlx/
  models/qwen-edit-2511-q6/
  loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
  ref.png
```

`ref.png` is ignored by git. Use your own private image. The default benchmark
assumes `576x768`; other aspect ratios work through `shardedit-edit`, but the
published benchmark comparisons use `576x768`.

## Model And LoRA Sources

The measured local runs used:

- Hugging Face model repo: [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux), folder `Qwen-Image-Edit-2511-q6`.
- LoRA repo: [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning), also mirrored on
  [ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning), file
  `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors`.

Why q6: local q4 tests struggled to preserve face identity, while q8 is larger
and was skipped because it was not worth the expected memory/runtime pressure
for this 24 GB M2 target. The tested mflux-ready q6 base weights were downloaded
from Hugging Face. Do not substitute an unrelated ModelScope package for this
asset. At setup time, other ModelScope resources were visible, including
2509-related or non-equivalent variants, but not this tested 2511 q6 mflux
folder.

Download helpers are optional:

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

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

After downloading, either rename/copy the q6 folder to
`models/qwen-edit-2511-q6` or export `SHARDEDIT_MODEL_PATH` to the actual folder.

## Dry Checks

Product entry:

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

Benchmark entry:

```bash
benchmarks/run_qwen_edit_benchmark.sh \
  --runtime shardedit \
  --dry-run
```

Dry-run prints the resolved model, LoRA, image, output size, speed preset, and
underlying mflux-compatible command. It does not run inference.

## First Real Run

```bash
mkdir -p outputs

shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

Use `--speed quality` first. It is slower than cache presets, but it is the
current fidelity path because it still runs every Transformer block.

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `image path does not exist: ref.png` | `ref.png` is not in the repo root | Put a private image at `ref.png` or pass `--image` |
| `model path does not exist` | model is not in `models/qwen-edit-2511-q6` | Move the model or export `SHARDEDIT_MODEL_PATH` |
| `lora path does not exist` | LoRA is not in `loras/` | Move the LoRA or export `SHARDEDIT_LORA_PATH` |
| very slow first run | cold model/Metal setup | Compare only same-machine, same-cooldown runs |
| output quality drifts in cache mode | F1B2 is approximate | Return to `--speed quality` and add manual review cases |

## Benchmark Discipline

When comparing presets, record:

- Mac model, chip, unified memory.
- macOS version.
- thermal state and cooldown.
- exact model and LoRA snapshot.
- output size, prompt, seed, steps, guidance.
- whether swap or memory pressure appeared.

The local 24 GB M2 numbers should be remeasured before applying them to 16 GB,
M2 Pro, M3, M4, or other hardware.
