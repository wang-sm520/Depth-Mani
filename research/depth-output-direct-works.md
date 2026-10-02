# 深度输出直接相关工作：DreamVLA / 3DDA / PAD

核查日期：2026-09-22。按主任务要求仅覆盖三篇。最终版收紧为实际取得 web 页面输出的证据；空返回不计确认。源码链接是动态 main，未固定 commit；未运行训练。删去初版未经最终页面输出复核的损失差异、配置和潜在 bug 断言。

## 总结

**“预测深度/未来深度帮助语言条件机器人策略”已有直接先例；不能据这三篇证明“全参数随机初始化、显式深度为唯一动作中间接口”已被完整做过。**

| 工作 | 公开时间 | 深度输入 / 输出 | 动作接口 | 从零判断 |
|---|---|---|---|---|
| PAD-Depth | 2024-11-27；NeurIPS 2024 | 当前深度输入 + 未来深度输出 | 多模态联合去噪 | 有预训练组件及预训练/适配流程，不是全模型随机初始化证据 |
| DreamVLA | 2025-07-06；核查 v3 | RGB 等输入，未来深度监督 | 动作读取潜在 world embedding；推理跳过深度解码 | 有随机 GPT 主干代码路径，但非全模型从零 |
| 3DDA / 3D Foresight | 首稿 2025-02-14；当前/未来深度任务明确见 2026-02-02 v2 | RGB 输入，当前深度和未来 RGB-D 输出 | 辅助监督，与动作共享模型 | scratch 指不做 cross-embodiment pretraining；源码仍加载预训练编码器 |

“从零”必须区分：①全部参数随机；②策略主干随机但编码器预训练；③不做额外机器人/跨本体预训练。不能互换。表中依据如下。

## DreamVLA

### 来源及时间

- [arXiv 2507.04447](https://arxiv.org/abs/2507.04447)：首稿 2025-07-06，v3 2025-08-26。
- [v3 全文](https://arxiv.org/html/2507.04447v3)，图 2、§3.1–3.3、§4.1、附录 A。
- [NeurIPS 2025 官方论文](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)。
- [官方代码](https://github.com/Zhangwenyao1/DreamVLA)。

### 深度输出与动作

图 2、§3.3：输入观测、语言、本体状态；专用 depth query 回归未来 `d_(t+n)`。有传感器时用真值，否则用 Depth-Anything 教师标签。图 2 明确预测头只训练使用；推理直接用 world embedding。**这是未来深度监督的潜表示，不是必须先解码深度图再生成动作。** 引言还明确单独使用深度或语义预测可能降低性能，不能将综合预测收益说成 depth-only 必然有效。[正文](https://arxiv.org/html/2507.04447v3)。

### 初始化

论文 §4.1 有机器人数据预训练再下游训练；图 2 明确冻结文本、视觉编码器。[正文](https://arxiv.org/html/2507.04447v3)。

[模型源码](https://raw.githubusercontent.com/Zhangwenyao1/DreamVLA/main/models/dreamvla_model.py)包含 `use_gpt2_pretrained=False`（148 行）；284 行起关闭时构造 GPT2 配置与模型，508–511 行开启时才调用 `from_pretrained`。**可确认随机策略主干路径存在，但论文各 run 实际选择哪个开关未确认；不能因模型叫 GPT-2 就认定一定加载 GPT-2 权重，也不能因此说全系统从零。**

## 3DDA / 3D Foresight

### 时间与版本边界

- [arXiv 历史](https://arxiv.org/abs/2502.10028)：v1 2025-02-14；v2 2026-02-02；v3 2026-03-05；v4 2026-03-26。当前标题为 *3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight*，标注 ICRA 2026。
- [v2 正文](https://arxiv.org/html/2502.10028v2)明确当前深度、未来 RGB-D、3D flow 三项任务；因此这些任务至少在 2026-02-02 已公开。**不能把当前版本方法不加区分地回填到 2025 首稿。**
- [v1 待逐项比较入口](https://arxiv.org/html/2502.10028v1)：初稿标题 ManiTrend 的线索需在总报告使用前单独核对；最终核查未完成 v1 全文输出头逐项排除。
- [v4 全文](https://arxiv.org/html/2502.10028v4)；[官方代码](https://github.com/Stardust-hyx/3D-Foresight)。

### 深度、动作及 scratch

当前版明确当前深度估计、未来 RGB-D 预测、3D flow 预测。它们是给策略加入 3D foresight 的自监督任务，不应误写成仅深度输入增强。[摘要](https://arxiv.org/abs/2502.10028)。

§III/§IV 将深度和未来 RGB-D 作为辅助预测，推理可删去辅助头。§IV-A 对 `3D Foresight (scratch)` 的精确定义是增强 GR-MG，但**不做 cross-embodiment pretraining**。CALVIN D→D scratch 为 4.01，去当前深度 3.97，去未来 RGB-D 3.94；这属于辅助目标消融，不是全随机模型证明。[v4 §IV-A、表 III](https://arxiv.org/html/2502.10028v4)。

[训练器源码](https://raw.githubusercontent.com/Stardust-hyx/3D-Foresight/main/policy/training/trainer_3D.py)加载预训练 MAE；114–119 行 `clip.load('ViT-B/32')` 并冻结。**公开流程依赖预训练编码器。** scratch 那一行是否仍加载完整 GR-MG checkpoint，未确认；总报告应避免擅自补全。

## PAD / PAD-Depth

### 来源及时间

- [arXiv 2411.18179](https://arxiv.org/abs/2411.18179)：首稿 2024-11-27，注明 NeurIPS 2024。
- [全文](https://arxiv.org/html/2411.18179v1)；[项目页](https://sites.google.com/view/pad-paper)；[官方代码](https://github.com/Robert-gyj/Prediction_with_Action)。

### 输入输出同时加深度

§3 的 Conditional Generation 明确条件含当前 RGB、pose、深度和语言，输出未来 RGB、未来深度及动作。§4.4 明确这是真机实验的 PAD-Depth；表 2 平均成功率 PAD 0.72、PAD-Depth 0.78。**因为输入和输出同时增加深度，不能将这 0.06 单独归因于深度输出 loss。** 深度被下采样成 `32×32×1` 并 patchify，不只是抽象深度 feature。[正文 §3、§4.4、附录 A.1.1](https://arxiv.org/html/2411.18179v1)。

### 初始化/源码

**更直接的初始化证据：§3.3 明确 PAD 从 ImageNet 图像生成预训练 DiT 初始化**，再修改不兼容的输入/输出层；§3.2 使用冻结预训练 VAE、冻结 CLIP。§4.1 再做 BridgeData-v2 200k steps 预训练、目标域 100k steps 适配。因此论文主实验甚至不属于“DiT 主干随机初始化”的正例。[正文 §3.2/§3.3/§4.1](https://arxiv.org/html/2411.18179v1)。

[train_robot.py](https://raw.githubusercontent.com/Robert-gyj/Prediction_with_Action/main/train_robot.py)中 `rgb_init` 非空则加载模型 checkpoint（266 行起）；训练入口还加载预训练 VAE。depth 分支提供 `depth_cond`、`depth` 与 `loss_depth`，支持论文“深度既输入又输出”的分类。**没有运行验证，完整 PAD-Depth 真机配置/权重能否直接复现未确认。**

## 给总报告的边界

1. 可以说“深度输出辅助策略已有直接先例”，不能说“depth-only 从零 VLA 已被这三篇完整实现”。
2. DreamVLA / 3D Foresight 更接近训练期深度辅助监督；PAD 更接近深度与动作联合生成。
3. 若用户目标是执行时可干预的显式深度中间接口，必须区别于推理时移除深度输出头的做法。
4. 未确认：全模块随机初始化实验；DreamVLA 各实验 GPT 初始化开关；3DDA scratch 完整 checkpoint 配置；PAD-Depth 真机复现完整性。

检索已使用 web，包含 DreamVLA/depth prediction、future depth robot policy、DepthVLA；最后者及其余工作按分工交主 agent，不在本文件扩展。
