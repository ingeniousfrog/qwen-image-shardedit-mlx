# Experiment Rationale

**Last Updated:** 2026-07-23

This file explains why qwen-image-shardedit-mlx has a conservative default, why many
experiment flags remain in the code, and why some promising micro-benchmarks
were not promoted.

## Fixed Scope

The current evidence covers:

| Item | Value |
| --- | --- |
| Hardware | base Apple M2, 24 GB unified memory |
| Workload | Qwen Image Edit, portrait reference, `576x768` output |
| Model | local `Qwen-Image-Edit-2511` q6 snapshot |
| LoRA | Lightning 8-step bf16 LoRA |
| Steps | 8 |
| Guidance | 1.0 |
| Seed | 42 |
| Training | none |

Other hardware and model snapshots need fresh benchmarks.

The base weights came from
[`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux),
using the `Qwen-Image-Edit-2511-q6` mflux-ready folder. The LoRA came from
[`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning),
which is also available on
[ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning),
using `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors`.

q6 was chosen because local q4 face-editing tests did not preserve identity
well enough, while q8 was larger and not worth the expected memory/runtime cost
for the 24 GB M2 target. The tested q6 base-weight package was downloaded from
Hugging Face; ModelScope was not used as the source for that exact mflux-ready
asset. At setup time, ModelScope had other visible resources, including
2509-related or non-equivalent variants, but not this tested 2511 q6 mflux
folder.

## Decision Table

| Experiment | Why try it | Evidence | Decision |
| --- | --- | --- | --- |
| Original mflux `--low-ram` | Establish the honest baseline | `1957.33s` wall at `576x768`, about `32m37s` | Baseline only |
| mflux without full low-RAM, cache-limited | Check whether flags alone fix the problem | First step projected about `36m20s` denoise | Rejected |
| qwen.image.swift | Test a narrower native Swift/MLX path | 256x256 smoke spent `259s` before failing on a missing tensor; GPL-3.0 also conflicts with MIT preference | Rejected as dependency |
| `guidance=1.0` branch pruning | Negative CFG branch has no effect when guidance is exactly 1.0 | Strictly equivalent optimization | Adopted |
| Stage materialization and timing events | Make MLX lazy execution measurable | Enables `SHARDEDIT_TIMING` logs and memory accounting | Adopted |
| Shard residency | Avoid keeping encoder and Transformer weights resident together | `291.31s` shell at historical 768x768; `367.94s` process at 576x768 fit-box | Default fidelity path on 24 GB |
| Window-4 residency | Explore lower-residency operation | Pixel-identical 768x768 output but slower at `379.40s` shell | Kept as constrained-memory candidate |
| Fit-box reference conditioning | Control reference token count across large photos | Keeps large references inside a comparable `576x768` box | Adopted |
| F1B2 residual cache | Skip middle Transformer blocks on safe-looking steps | `232.66s` process at 576x768, but visible softening/drift risk remains | Opt-in speed preset |
| Flow-aware cache threshold | Make cache decisions timestep-aware | `258.24s` process at 576x768, more conservative than fixed F1B2 | Opt-in balanced preset |
| TaylorSeer-style predictors | Predict future residuals instead of copying old ones | Infrastructure exists; not yet proven on 8-step multi-reference quality gates | Research only |
| Token merge V0 | Reduce token count inside full misses | Text-only did not speed up; condition/both hurt face quality | Rejected, kept for audit |
| Dense image MLP | Use dense bf16 for image feed-forward layers | Micro/sweep wins, but end-to-end F1B2+dense was slower | Diagnostic only |
| Image QKV fusion | Reduce projection overhead | `0.987x` short-run and `1.008x` long-run median | Rejected |
| K-quant image MLP | Test better q6 kernel throughput | Micro path reached `1.381x`, but e2e was slower and produced an all-black output | Diagnostic/no-go |
| Custom q6 Metal kernels | Try application-level kernel replacement | Fork/tiled and simdgroup paths were slower than MLX eager | Archived |

## Why Shard Is The Default

Shard residency changes weight lifecycle, not model math. The no-cache path
still executes all 8 denoise steps and all 60 Transformer blocks per step.
That makes it slower than F1B2, but safer as a public default.

The current speedup over original mflux comes mainly from:

- avoiding useless negative branch work at `guidance=1.0`;
- avoiding resident-weight memory cliffs;
- using shard window loading for the q6 Transformer;
- capping reference conditioning with fit-box.

## Why Cache Is Opt-In

F1B2 can make cache-hit steps much faster because it computes the first block
and last two blocks, then reuses a previous middle residual. That is an
approximation. It can soften hair, clothing texture, and face details even when
coarse MAE/PSNR thresholds pass.

The current rule is:

```text
machine pixel gate qualifies a candidate for review;
manual identity and detail review decides whether it can be promoted.
```

## Quality Gate

The local quality work used a six-case `576x768` fit-box matrix and passed the
coarse automated coverage gate:

| Metric | Result |
| --- | ---: |
| comparable cases | 6 |
| failed pixel cases | 0 |
| mean MAE | 7.58 |
| mean PSNR | 26.81 dB |
| mean changed channel ratio | 0.9335 |

That is not enough for a default cache preset. Face identity, hair edges,
clothing texture, pose stability, and prompt adherence still need human review.

The open-source manifest at `benchmarks/quality_cases.json` is a sanitized
starter. Replace it with private references and generated baseline/candidate
outputs when evaluating your own hardware.

## Hardware Caveat

The 24 GB M2 result should not be projected directly to 16 GB Macs or faster
chips. For every new machine, rerun:

```bash
benchmarks/run_qwen_edit_benchmark.sh \
  --runtime shardedit \
  --width 576 \
  --height 768 \
  --steps 8 \
  --reference-conditioning-size fit-box \
  --reference-conditioning-max-width 576 \
  --reference-conditioning-max-height 768 \
  --cache-preset none
```

Then compare cache presets only after the no-cache baseline is stable.
