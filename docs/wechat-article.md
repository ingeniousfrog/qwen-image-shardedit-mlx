# 在“老破小”里跑 Qwen Image Edit：别硬装挑空客厅

更新日期：2026-07-23

我这台 MacBook Pro 很适合拿来做一个不太体面的比喻：它像一套“老破小”。

不是不能住。地段还行，Apple Silicon，MLX，统一内存，底子并不差。但你真想往里塞一套 Qwen Image Edit，再加 q6 权重、Vision/Text Encoder、VAE、Lightning LoRA，就像在老房子里硬装挑空客厅：效果图很漂亮，施工队一进门先量层高，然后大家都沉默了。

原计划当然是“高端大气”：本地跑图，不上传照片；人像编辑，尽量保脸；Lightning 8-step，速度别太离谱；最好还能把 30 分钟压到 1 分钟。听上去像大平层改造。

实际情况是：这套房只有 24 GB 统一内存。

## 先验房：这不是豪宅

当前这轮数据只对应一台机器：

| 项目 | 设置 |
| --- | --- |
| 机器 | Apple M2，24 GB unified memory |
| 模型 | `fcreait/Qwen-Image-Edit-mflux` 里的 Qwen-Image-Edit-2511 q6 |
| LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` 的 8-step bf16，Hugging Face / ModelScope 都有 |
| 输出 | 576x768 |
| steps / guidance / seed | 8 / 1.0 / 42 |
| 训练 | 没有训练、没有蒸馏、没有重新量化 |

这点必须先说清楚。不同芯片、不同内存、不同模型快照，都要重新验房。把 24 GB M2 的数据直接搬到 16 GB 或 M4 上，就像拿隔壁小区的户型图指导自家装修，容易翻车。

模型本身也不小：Transformer 大约 15.5 GiB，Text/Vision Encoder 大约 14.4 GiB，VAE 反而只有两百多 MiB。24 GB 不是不能跑，而是不能让所有大家具一起摆在客厅中央。

权重我最后选了 q6。q4 倒是省地方，但人脸保持比较吃力；q8 更大，我也没耐心下载，而且在 24 GB M2 上大概率会让内存压力更难看。q6 不是豪装，也不是出租屋配置，算是这套老房子能承受的中档材料。

这里还有个容易踩坑的地方：这份 q6 基础权重是在 Hugging Face 的 `fcreait/Qwen-Image-Edit-mflux` 下载的，不是 ModelScope。准备实验时，我没有在 ModelScope 找到这份 mflux-ready 2511 q6 权重；能看到的是 2509 相关资源或其他不等价的变体。LoRA 倒是两边都有，Hugging Face 和 ModelScope 都能拿到 `lightx2v/Qwen-Image-Edit-2511-Lightning`。

## 毛坯基线：32 分钟

先跑原始 mflux `--low-ram`，别急着优化。结果很朴素：

| 路径 | 时间 |
| --- | ---: |
| 原始 mflux `--low-ram` | 1957.33s wall，约 32m37s |

这就是毛坯房。能住，但你每改一次 prompt 都要出去吃顿饭再回来。

我也试过“不走完整 low-RAM，只给 MLX cache 限额”这种办法。结果第一步就投影到三十多分钟，没有本质改善。结论很直接：这不是拧几个运行参数就能解决的事，真正的问题是内存驻留和 Transformer full pass。

## 第一刀：拆掉非承重墙

最干净的优化是 `guidance=1.0`。

CFG 公式里，当 guidance 正好等于 1，负分支对结果没有贡献。原来还去算正负两条分支，就像明知道那面墙不是承重墙，还非要让它占地方。

所以 qwen-image-shardedit-mlx 先把这块剪掉。这个优化不改变数学结果，属于可以放心拆的非承重墙。

接着做的是阶段边界和日志：哪里在 load，哪里在 compute，哪里在 release，峰值内存是多少，都写进 timing event。没有这些记录，优化就会变成“我感觉客厅变大了”。工程上不能这么装修。

## 真正的改造：做收纳，而不是扩建

`shard` residency 是现在的默认保真路径。

它不跳过模型，也不减少 Transformer block。8 step x 60 block，480 个 block 仍然完整执行。它做的是“收纳系统”：Transformer 权重按窗口进入 MLX，用完释放或换下一组，不让 Encoder、Transformer、VAE 和激活一起挤爆统一内存。

效果是：

| 路径 | 时间 | 判断 |
| --- | ---: | --- |
| qwen-image-shardedit-mlx shard no-cache | 367.94s process，约 6m08s | 当前默认保真路径 |

从 32 分钟到 6 分钟，不是因为这套老房子突然变成豪宅，而是东西终于各归各位了。

## 层高不够，就别硬装水晶吊灯

更快的办法也有：F1B2 residual cache。

F1B2 命中时只算第 1 个 Transformer block 和最后 2 个 block，中间 57 个 block 复用上一轮 residual。速度确实漂亮：

| 路径 | 时间 | 判断 |
| --- | ---: | --- |
| shard + F1B2/max=1 | 232.66s process，约 3m53s | opt-in 速度档 |
| shard + flow-aware | 258.24s process，约 4m18s | opt-in 保真向 cache |

但人像编辑最怕的不是整图炸掉，而是脸慢慢不像，发丝变软，衣服纹理被抹平。机器指标可以过，肉眼还是会不舒服。

所以 cache 在这个项目里只能是选配。你可以自己试，但默认不打开。老破小也能住得舒服，前提是别为了显大把承重结构改没了。

## 为什么有些“失败方案”还留着

代码里有不少看起来很复杂、甚至已经被放弃的参数。我没有删，因为这些是施工记录。

| 方案 | 结果 | 为什么不删 |
| --- | --- | --- |
| Swift/qwen.image.swift | 兼容性、性能、许可证都不合适 | 说明为什么不换底座 |
| token merge V0 | text-only 不提速，condition/both 伤脸 | 防止以后重复拆错墙 |
| dense image MLP | micro benchmark 好看，端到端变慢 | 局部建材好，不等于整屋好 |
| K-quant image MLP | micro 有收益，e2e 更慢且出黑图 | 这是明确 no-go 证据 |
| 自定义 q6 Metal kernel | 比 MLX eager 慢 | 暂时不自研硬装件 |

这些参数不适合普通用户第一次运行时看见。开源版把入口收窄成 `image / prompt / clarity / speed / seed`，研究参数放进文档。常规使用别碰工具间；想做实验，再自己打开。

## 开源版怎么收拾

项目改名成 `qwen-image-shardedit-mlx`，默认约定也重新整理过：

- 默认测试图：`ref.png`
- 默认输出尺寸：`576x768`
- 默认模型目录：`models/qwen-edit-2511-q6`
- 默认 LoRA 目录：`loras/...safetensors`
- 默认速度：`quality`，也就是 shard + no-cache
- 协议：MIT

仓库不放真实照片，不放模型权重，不放 LoRA，不放生成图。用户把自己的 `ref.png` 放进去，先 dry-run，再跑：

```bash
shardedit-edit \
  --image ref.png \
  --prompt "Replace the background with a naturally lit cafe interior while preserving identity." \
  --speed quality
```

如果要试速度，再看 `balanced` 和 `fast`。如果要复现实验，再看 `docs/parameters.md` 和 `docs/experiment-rationale.md`。

## 现在的结论

这轮改造不是把老破小装修成空中别墅。

更像是：量清楚层高，确认哪些墙能拆，哪些墙不能碰；该收起来的收起来，该腾出来的腾出来；先让它稳定、可复现、能住人。32 分钟到 4 到 6 分钟，是一个实用进展。1 分钟还没到，也不能装作已经到了。

本地 AI 图像编辑真正难的地方不只是“快”，而是快得有证据，质量有门槛，失败方案也讲得清楚。不装豪宅，先把基础工程做好。
