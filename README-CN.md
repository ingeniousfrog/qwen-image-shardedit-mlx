# qwen-image-shardedit-mlx

**更新日期：** 2026-07-23

qwen-image-shardedit-mlx 是一个非官方的 Apple Silicon / MLX 运行时实验项目，目标是让
Qwen Image Edit 在本地 Mac 上更可用。它保留 mflux 兼容路径，加入 shard residency
来降低 24 GB 统一内存上的驻留压力，并把 residual cache 作为可选实验档暴露出来。

仓库不包含模型权重、LoRA、真实参考图或生成结果。默认测试图名是 `ref.png`，默认
benchmark 尺寸是 `576x768`。

## 当前状态

当前实测数据来自一台基础版 Apple M2、24 GB 统一内存的 Mac。它能说明这台机器上的
工程取舍，不能直接当成其他硬件的成绩。

| 路径 | 输出 | M2 24 GB 实测 | 判断 |
| --- | --- | ---: | --- |
| 原始 mflux `--low-ram` | 576x768 | 1957.33s wall，约 32m37s | 基线 |
| shard，无 cache | 576x768 fit-box | 367.94s process，约 6m08s | 默认保真路径 |
| shard + F1B2/max=1 | 576x768 fit-box | 232.66s process，约 3m53s | opt-in 速度档 |
| shard + flow-aware | 576x768 fit-box | 258.24s process，约 4m18s | opt-in 保真向 cache |

一分钟还没到。Cache 档虽然通过了当前粗 pixel screen，但人脸身份、发丝、服装纹理等仍然
需要人工复核，所以默认不打开 cache。

## 依赖

普通运行需要：

- macOS + Apple Silicon，且当前终端能正常访问 Metal/MLX。
- Python 3.11 或更新。
- `mflux>=0.18`
- `mlx>=0.31`
- `safetensors>=0.8`
- `Pillow>=10`
- 本地 Qwen Image Edit q6 模型。
- 本地 Lightning LoRA。
- 一张你自己的私有参考图：`ref.png`。

`mflux` 是必需的。qwen-image-shardedit-mlx 目前是 mflux-compatible layer，不是从零写的完整
Qwen Image Edit runtime。

`mlx-kquant` 不是普通运行必需项。只有复现 K-quant image MLP 诊断实验时才需要安装。

Xcode / `metal` / `metallib` 也不是默认跑图必需项。它们只在复现被放弃的 Swift
兼容路径，或做自定义 Metal kernel 实验时需要。

## 安装

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[runtime,test]"
```

如果要复现 K-quant 诊断实验，再装：

```bash
python -m pip install -e ".[experiments]"
```

## 默认文件结构

```text
qwen-image-shardedit-mlx/
  models/qwen-edit-2511-q6/
  loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
  ref.png
```

`ref.png`、`models/`、`loras/`、`outputs/`、`benchmark-runs/` 都不应该提交到公开仓库。

## 模型和 LoRA 来源

当前实测用的是：

- 基础模型权重：Hugging Face 上的 [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux)，具体使用 `Qwen-Image-Edit-2511-q6`。
- LoRA：[`lightx2v/Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)，Hugging Face 和
  [ModelScope](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning) 上都有；这里具体使用
  `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors`。

选 q6 是一个折中：q4 本地跑人像时脸比较难保持住；q8 文件更大，我没有下载，而且在
24 GB M2 上也更可能增加内存和运行压力。q6 不是最省，也不是最满，但比较适合这轮
“老破小”实验。注意，实测的这份 mflux-ready q6 基础权重是在 Hugging Face 下载的；
ModelScope 不能当作这份 q6 权重的来源。准备这轮实验时，我没有在 ModelScope 找到这份
mflux-ready 2511 q6 权重；能看到的是 2509 相关资源或其他不等价的变体。

下载工具不是运行时依赖，可以按需安装：

```bash
python -m pip install "huggingface_hub[hf_xet]" modelscope

# 下载 q6 mflux-ready 权重目录。下载后可以重命名为 models/qwen-edit-2511-q6，
# 也可以把 SHARDEDIT_MODEL_PATH 指向实际下载目录。
huggingface-cli download \
  fcreait/Qwen-Image-Edit-mflux \
  Qwen-Image-Edit-2511-q6 \
  --local-dir models/qwen-edit-mflux

# LoRA 下载二选一。ModelScope：
modelscope download \
  --model lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local_dir loras

# 或 Hugging Face：
huggingface-cli download \
  lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  --local-dir loras
```

如果你把模型或 LoRA 放在别处，可以传参数，也可以设置环境变量：

```bash
export SHARDEDIT_MODEL_PATH=/path/to/qwen-edit-2511-q6
export SHARDEDIT_LORA_PATH=/path/to/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
export SHARDEDIT_IMAGE_PATH=/path/to/ref.png
```

## 先 Dry Run

先确认命令会如何展开，不跑真实推理：

```bash
shardedit-edit \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --dry-run
```

Benchmark harness 也可以 dry-run：

```bash
benchmarks/run_qwen_edit_benchmark.sh \
  --runtime shardedit \
  --dry-run
```

Dry-run 会打印模型、LoRA、图片、输出尺寸、speed preset 和底层 mflux-compatible 命令。

## 第一次真实运行

建议先用保真路径：

```bash
mkdir -p outputs

shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality \
  --output outputs/ref-cafe.png
```

`--speed quality` 会走 shard residency + no cache。它比 cache 档慢，但仍然完整计算每个
Transformer block，是当前默认保真路径。

## 常规参数和实验参数

常规用户先用 `shardedit-edit`：

| 参数 | 含义 |
| --- | --- |
| `--image` | 参考图，默认 `ref.png` |
| `--prompt` | 编辑指令 |
| `--clarity standard` | 输出放进 `768x768` box；`576x768` 参考图会保持 `576x768` |
| `--speed quality` | shard residency，无 residual cache |
| `--speed balanced` | flow-aware F1B2 cache，选配 |
| `--speed fast` | fixed F1B2 cache，选配 |
| `--seed` | 随机种子，默认 `42` |

研究参数集中在 `benchmarks/run_qwen_edit_benchmark.sh` 和 `--shardedit-*` flags。
token merge V0、dense image MLP、K-quant image MLP、自定义 q6 Metal kernel 等路径
保留在代码里，是为了复现实验和说明为什么没有采用，不是推荐默认开关。

完整参数见 [docs/parameters.md](docs/parameters.md)。实验取舍见
[docs/experiment-rationale.md](docs/experiment-rationale.md)。公众号草稿见
[docs/wechat-article.md](docs/wechat-article.md)。

## 发布前检查

```bash
rg -n "lxy-1|/Users/|/private/tmp|benchmark-runs/2026|\\.safetensors" .
git status --short
```

命中应该只剩文档里的检查命令本身，不能有真实图片路径、本机绝对路径或生成结果。

代码采用 MIT 协议。模型权重、LoRA、mflux、MLX 和 Qwen 相关资产各自遵循它们自己的协议。
