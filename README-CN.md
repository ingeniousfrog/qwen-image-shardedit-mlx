# qwen-image-shardedit-mlx

[English](README.md)

面向 Apple Silicon / MLX 的非官方 [Qwen Image Edit](https://github.com/QwenLM/Qwen-Image) 运行时，以 mflux 兼容层的形式交付。核心能力是 **shard residency**：在 24 GB 统一内存上把完整的 8 步 × 60 block 编辑路径跑通，避开原版 `--low-ram` 的墙钟悬崖；residual cache 则以 **可选** 速度档暴露，默认关闭。

> 仓库只包含代码与文档，不含模型权重、LoRA、参考图或生成结果。

## 特性

- **Shard residency** — 按块流式驻留，降低整栈常驻压力
- **产品向 CLI** — `shardedit-edit` 将 clarity / speed / seed 映射为底层运行时参数
- **保守默认** — `quality` = shard + 无 residual cache（`guidance=1.0` 下完整计算每个 block）
- **可选 cache 档** — `balanced` / `fast` 用近似换时间；未人工复核人脸前请勿当作默认
- **可复现证据** — 保留 benchmark harness、取舍说明与实验开关，便于审计

## 实测结果

硬件：基础版 Apple **M2**，**24 GB** 统一内存。人像参考图 → `576×768` fit-box。数字说明本机工程取舍，不能直接外推到其他机器。

| 路径 | 耗时（process） | 角色 |
| --- | ---: | --- |
| 原版 mflux `--low-ram` | 约 32m 37s（wall） | 基线 |
| shard，无 cache | 约 6m 08s | **默认保真路径** |
| shard + F1B2 / max=1 | 约 3m 53s | 可选速度档 |
| shard + flow-aware cache | 约 4m 18s | 可选保真向 cache |

**未**声称一分钟内完成。Cache 档可通过粗 pixel screen，但人脸身份、发丝、服装等仍需人工复核后再决定是否默认开启。

## 环境要求

| 项 | 说明 |
| --- | --- |
| 硬件 | macOS + Apple Silicon，终端可正常使用 Metal / MLX |
| Python | 3.11+ |
| 运行时依赖 | `mflux≥0.18`、`mlx≥0.31`、`safetensors≥0.8`、`Pillow≥10` |
| 本地资产 | Qwen Image Edit **q6** 权重、Lightning LoRA、私有 `ref.png` |

`mlx-kquant` 与完整 Xcode / Metal 命令行工具为可选项，仅用于复现已放弃的 kernel / K-quant 探针。详见 [docs/installation.md](docs/installation.md)。

本项目是 mflux 兼容层，不是从零实现的完整 Qwen Image Edit runtime；`mflux` 为必需依赖。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

如需 K-quant 诊断实验：

```bash
python -m pip install -e ".[experiments]"
```

建议本地布局（均已 gitignore）：

```text
models/qwen-edit-2511-q6/
loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
ref.png
```

可用参数或环境变量覆盖路径（`SHARDEDIT_MODEL_PATH`、`SHARDEDIT_LORA_PATH`、`SHARDEDIT_IMAGE_PATH`）。可参考 [.env.example](.env.example)。

## 模型与 LoRA

实测使用：

| 资产 | 来源 | 文件 / 目录 |
| --- | --- | --- |
| 基础权重（q6） | [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux) | `Qwen-Image-Edit-2511-q6` |
| Lightning LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)（[ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)） | `…-8steps-V1.0-bf16.safetensors` |

**为何选 q6：** 本地 q4 人像编辑身份保持偏弱；q8 体积更大，且在 24 GB M2 上预期内存/耗时压力更高。实测的 mflux-ready q6 包来自 Hugging Face；不要假设 ModelScope 上有同一目录（可见资源可能是 2509 或其他不等价变体）。

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux
# 重命名/软链到 models/qwen-edit-2511-q6，或设置 SHARDEDIT_MODEL_PATH

# LoRA：ModelScope 或 Hugging Face 二选一
modelscope download \
  --model lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local_dir loras
```

## 快速开始

仅检查命令映射（不推理）：

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

建议首次真实运行走保真路径：

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

Benchmark harness dry-run：

```bash
benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --dry-run
```

### 常用参数

| 参数 | 含义 |
| --- | --- |
| `--clarity standard` | 输出落入 `768×768` box；`576×768` 参考图保持原尺寸 |
| `--speed quality` | shard residency，**无** residual cache（默认） |
| `--speed balanced` | flow-aware F1B2 cache（可选） |
| `--speed fast` | fixed F1B2 cache（可选） |
| `--seed` | 随机种子（默认 `42`） |

底层 `--shardedit-*` 与已放弃探针（token merge V0、dense / K-quant image MLP、自定义 q6 Metal kernel）仍保留在代码中便于复现，**不是**推荐默认开关。完整参数见 [docs/parameters.md](docs/parameters.md)。

## 设计取舍

默认偏保守：

1. `guidance=1.0` 下的剪枝在数学上精确。
2. Shard residency 仍执行全部 8 步 × 60 Transformer block。
3. Fit-box 条件化控制参考图 token 预算。
4. Residual cache 为近似路径 → 人工复核人脸通过前保持 opt-in。

实验史与取舍说明：[docs/experiment-rationale.md](docs/experiment-rationale.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/installation.md](docs/installation.md) | 依赖分层、Metal 说明、首次运行 |
| [docs/parameters.md](docs/parameters.md) | 产品 CLI 与实验参数 |
| [docs/experiment-rationale.md](docs/experiment-rationale.md) | 尝试 / 保留 / 放弃的决策 |
| [docs/open-source-checklist.md](docs/open-source-checklist.md) | 发布前卫生检查 |

## 许可证

项目代码采用 **MIT**。模型权重、LoRA、mflux、MLX 与 Qwen 相关资产遵循各自协议。请勿将私有图片、权重、LoRA 或生成结果提交到公开仓库。
