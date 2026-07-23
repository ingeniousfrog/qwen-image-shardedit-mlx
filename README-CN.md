# qwen-image-shardedit-mlx

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English README](README.md)

qwen-image-shardedit-mlx 是一个面向 Apple Silicon / MLX 的非官方
[Qwen Image Edit](https://github.com/QwenLM/Qwen-Image) 运行时。它保留
mflux 兼容路径，加入 shard residency 以降低 24 GB 统一内存上的驻留压力，并把
residual cache 作为可选速度档暴露出来。安装后提供命令行工具 `shardedit-edit`。

仓库只包含代码和文档，不含模型权重、LoRA、参考图或生成结果。

## 功能

- 通过 mflux 兼容层在 Apple Silicon 上本地跑 Qwen Image Edit。
- 用 shard residency 按块流式驻留，而不是把整栈常驻在统一内存里。
- 用 `shardedit-edit` 把 `clarity` / `speed` / `seed` 映射成底层运行时参数。
- 默认走保守路径：shard residency，且不启用 residual cache。
- 需要更快时，可选用 `balanced` 或 `fast` residual-cache 档。
- 用 fit-box 条件化控制参考图 token 预算（`576x768` 会保持 `576x768`）。
- 支持 dry-run，先确认命令映射再花时间推理。
- 保留已放弃探针，方便复现实验；它们不是推荐默认开关。

## 实测结果

硬件：基础版 Apple M2，24 GB 统一内存。任务：人像参考图、`576x768` fit-box、
Lightning 8-step LoRA。这些数字说明本机工程取舍，不能直接当成其他硬件的成绩。

| 路径 | 耗时 | 角色 |
| --- | ---: | --- |
| 原版 mflux `--low-ram` | 约 32m 37s wall | 基线 |
| shard，无 cache | 约 6m 08s process | 默认保真路径 |
| shard + F1B2 / max=1 | 约 3m 53s process | 可选速度档 |
| shard + flow-aware cache | 约 4m 18s process | 可选保真向 cache |

未声称一分钟内完成。Cache 档可通过粗 pixel screen，但人脸身份、发丝、服装等仍
需人工复核后再决定是否默认开启。

## 安装

创建虚拟环境并安装运行时依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

仅在复现 K-quant 诊断实验时再装：

```bash
python -m pip install -e ".[experiments]"
```

运行时 Python 依赖是 `mflux>=0.18`、`mlx>=0.31`、`safetensors>=0.8`、`Pillow>=10`。
`mflux` 是必需的：本项目是窄的 mflux 兼容层，不是从零实现的完整 Qwen Image Edit
runtime。

`mlx-kquant` 和完整 Xcode / Metal 命令行工具是可选项，只用于复现已放弃的 kernel
或 K-quant 探针。依赖分层见 [docs/installation.md](docs/installation.md)。

建议本地布局（均已 gitignore）：

```text
models/qwen-edit-2511-q6/
loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
ref.png
```

可用 CLI 参数或环境变量覆盖路径
（`SHARDEDIT_MODEL_PATH`、`SHARDEDIT_LORA_PATH`、`SHARDEDIT_IMAGE_PATH`）。
常见变量见 [.env.example](.env.example)。

## 模型与 LoRA

实测使用：

| 资产 | 来源 | 文件 / 目录 |
| --- | --- | --- |
| 基础权重（q6） | [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux) | `Qwen-Image-Edit-2511-q6` |
| Lightning LoRA | [`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)（[ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)） | `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` |

选 q6 是折中：本地 q4 人像编辑身份保持偏弱；q8 体积更大，且在 24 GB M2 上预期
内存/耗时压力更高。实测的 mflux-ready q6 包来自 Hugging Face；不要假设
ModelScope 上有同一目录——可见资源可能是 2509 或其他不等价变体。

下载示例：

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux

# LoRA：ModelScope
modelscope download \
  --model lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local_dir loras

# 或 Hugging Face
huggingface-cli download \
  lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local-dir loras
```

把 q6 目录重命名/软链到 `models/qwen-edit-2511-q6`，或把
`SHARDEDIT_MODEL_PATH` 指到实际下载路径。

## CLI 用法

这里的 `shardedit-edit` 不是 mflux 原生命令，而是本项目安装时注册的 console
script。来源在 `pyproject.toml`：

```toml
[project.scripts]
shardedit-edit = "shardedit_mlx.product_cli:main"
```

它是普通用户入口：接收 `image` / `prompt` / `speed` / `seed` 等参数，再展开成
底层 `shardedit_mlx.mflux_fast_edit` 和 mflux 运行参数。

先 dry-run，确认命令如何展开：

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

第一次真实运行建议走保真路径：

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

人工复核身份后再试可选 cache 档：

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed balanced \
  --output outputs/ref-cafe-balanced.png
```

Benchmark harness dry-run：

```bash
benchmarks/run_qwen_edit_benchmark.sh --runtime shardedit --dry-run
```

常用参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--image` | `ref.png` | 参考图，或逗号分隔多图 |
| `--prompt` | 必填 | 编辑指令 |
| `--clarity standard` | `standard` | 输出落入 `768x768` box，保持比例 |
| `--clarity high` | 关 | 输出落入 `1024x1024` box |
| `--speed quality` | `quality` | shard residency，无 residual cache |
| `--speed balanced` | 关 | flow-aware F1B2 cache |
| `--speed fast` | 关 | fixed F1B2 cache |
| `--seed` | `42` | 随机种子 |
| `--dry-run` | 关 | 只打印映射，不推理 |

底层 `--shardedit-*` 与已放弃探针（token merge V0、dense / K-quant image MLP、
自定义 q6 Metal kernel）仍保留在代码里便于审计，不是推荐默认开关。完整参数见
[docs/parameters.md](docs/parameters.md)。

## Speed Preset

| Preset | 运行形态 | 状态 |
| --- | --- | --- |
| `quality` | `shard`，无 cache，fit-box 条件化 | 默认保真路径 |
| `balanced` | `shard`，F1B2/max=1，flow-aware 阈值 | 可选 cache 候选 |
| `fast` | `shard`，F1B2/max=1，fixed 阈值 | 可选速度候选 |

默认偏保守：

- `guidance=1.0` 下的剪枝在数学上精确。
- Shard residency 仍执行全部 8 步 × 60 Transformer block。
- Fit-box 条件化控制大参考图的 token 预算。
- Residual cache 是近似路径，人工复核人脸通过前保持 opt-in。

实验史与取舍说明见 [docs/experiment-rationale.md](docs/experiment-rationale.md)。

## 开发

```bash
python -m pip install -e ".[runtime,test]"
python -m pytest
```

更多文档：

| 文档 | 内容 |
| --- | --- |
| [docs/installation.md](docs/installation.md) | 依赖分层、Metal 说明、首次运行 |
| [docs/parameters.md](docs/parameters.md) | 产品 CLI 与实验参数 |
| [docs/experiment-rationale.md](docs/experiment-rationale.md) | 尝试 / 保留 / 放弃的决策 |
| [docs/open-source-checklist.md](docs/open-source-checklist.md) | 发布前卫生检查 |

## 许可证

项目代码采用 MIT。模型权重、LoRA、mflux、MLX 与 Qwen 相关资产遵循各自协议。
请勿将私有图片、权重、LoRA 或生成结果提交到公开仓库。
