# Open-Source Checklist

**Last Updated:** 2026-07-23

Run this before copying the folder to a public repository.

## Data Hygiene

- `ref.png` is ignored and should be a private local file only.
- `models/` and `loras/` are ignored.
- `benchmark-runs/`, `outputs/`, and `swift-overlays/` are ignored.
- No generated output images are included.
- No local absolute paths are used as defaults.

Check:

```bash
rg -n "lxy-1|/Users/|/private/tmp|benchmark-runs/2026|IMG_|\\.safetensors" .
git status --short
```

Expected hits should be only generic model/LoRA names in docs or examples, not
private file paths or bundled assets.

## License

- Project code: MIT.
- mflux, MLX, model weights, LoRA files, and Qwen assets keep their own licenses.
- Do not copy GPL source into this project. The Swift experiment is documented
  as rejected evidence only.

## User-Facing Defaults

- Project name: `qwen-image-shardedit-mlx`.
- Python package: `shardedit_mlx`.
- CLI: `shardedit-edit`, `shardedit-mflux-edit`, `shardedit-warm-edit`.
- Default image: `ref.png`.
- Default benchmark size: `576x768`.
- Default fidelity path: `shard` residency, fit-box conditioning, no residual cache.

## Release Notes To Keep Honest

- Say the measured machine was Apple M2 with 24 GB unified memory.
- Say one minute has not been reached.
- Say cache presets are opt-in and need manual face review.
- Say other hardware must rerun benchmarks.
- Say no model weights, LoRAs, or sample portraits are bundled.
