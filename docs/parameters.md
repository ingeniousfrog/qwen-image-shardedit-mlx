# Parameter Guide

**Last Updated:** 2026-07-23

qwen-image-shardedit-mlx keeps many experiment flags for reproducibility, but normal users
should start with the product-facing CLI.

## Normal Use

Use `shardedit-edit`:

| Option | Default | Use |
| --- | --- | --- |
| `--image` | `ref.png` | One reference image, or comma-separated image paths |
| `--prompt` | required | Edit instruction |
| `--clarity standard` | `standard` | Fit output into a `768x768` box while preserving aspect ratio |
| `--clarity high` | off | Fit output into a `1024x1024` box |
| `--speed quality` | `quality` | Shard residency, no residual cache |
| `--speed balanced` | off | Flow-aware F1B2 cache, opt-in |
| `--speed fast` | off | Fixed F1B2 cache, opt-in |
| `--seed` | `42` | Reproducible seed |
| `--lightning-steps` | `8` | Choose 8-step or 4-step Lightning LoRA |
| `--model` | `models/qwen-edit-2511-q6` | Local model directory |
| `--lora` | matching Lightning LoRA path | Explicit LoRA override |
| `--dry-run` | off | Print mapping without running inference |

With a `576x768` `ref.png`, `--clarity standard` resolves to `576x768`.

## Runtime Presets

| Preset | Runtime shape | Status |
| --- | --- | --- |
| `quality` | `shard`, no cache, fit-box conditioning | default fidelity path |
| `balanced` | `shard`, F1B2/max=1, flow-aware threshold | opt-in cache candidate |
| `fast` | `shard`, F1B2/max=1, fixed threshold | opt-in speed candidate |

Cache presets can be useful, but they are approximate. Pixel metrics only catch
large failures; manual face, hair, clothing, and prompt-adherence checks still
decide whether a preset is acceptable.

## Benchmark Harness

Use `benchmarks/run_qwen_edit_benchmark.sh` when reproducing evidence:

```bash
benchmarks/run_qwen_edit_benchmark.sh \
  --runtime shardedit \
  --width 576 \
  --height 768 \
  --steps 8 \
  --guidance 1.0 \
  --reference-conditioning-size fit-box \
  --reference-conditioning-max-width 576 \
  --reference-conditioning-max-height 768
```

The harness writes command, stdout, stderr, timing, environment, memory, and
thermal notes into `benchmark-runs/`.

## Experiment Flag Groups

| Group | Examples | Use |
| --- | --- | --- |
| Residency | `--residency shard`, `--residency window`, `--release-policy` | Memory lifecycle experiments |
| Conditioning | `--reference-conditioning-size fit-box`, `--reference-conditioning-max-width 576` | Control reference token count |
| Residual cache | `--cache-preset f1b2`, `--cache-threshold`, `--cache-back-blocks` | Approximate acceleration |
| Flow-aware cache | `--cache-threshold-schedule flow-aware` | Timestep-aware cache decisions |
| Predictor probes | `--cache-predictor linear`, `--cache-predictor adams-bashforth` | TaylorSeer-style experiments |
| Diagnostics | `--q6-linear-profile`, `--probe-blocks`, `--token-redundancy-blocks` | Measurement only |
| Rejected probes | `--condition-token-merge`, `--text-token-merge`, `--dense-img-ff-window`, `--kquant-img-ff-window` | Kept for audit/reproduction, not default paths |

## Recommended Defaults

For a first run:

```bash
shardedit-edit --image ref.png --prompt "..." --speed quality
```

For local speed exploration after checking the quality baseline:

```bash
shardedit-edit --image ref.png --prompt "..." --speed balanced
```

For research reproduction:

```bash
benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --cache-preset f1b2
```

Do not promote a cache or diagnostic flag just because one run is faster. Use
same-machine A/B outputs, the quality manifest, and manual review.
