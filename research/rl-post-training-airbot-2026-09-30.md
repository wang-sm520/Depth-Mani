# AIRBOT 深度策略用 RL 后训练：方法调研与实施路线

调研日期：2026-09-30。范围是当前仓库中的 AIRBOT 深度行为克隆策略、LIBERO/robosuite 仿真，以及把已有示范策略继续用强化学习优化的可行路线。优先引用算法论文、作者代码和环境官方仓库；本文是针对当前工程的落地调研，不是通用 RL 教程。

## 结论先行

当前最合适的路线是 **先把 BC 策略变成受约束的 residual policy，在 LIBERO 或可靠的 AIRBOT 仿真中做 offline-to-online 训练；首个 offline baseline 选 IQL 或 TD3+BC，首个在线 baseline 选 residual SAC/TD3**。不要直接把 6.67M 参数的确定性视觉策略交给 PPO 在真机上探索。

原因有三点：

1. 现有 AIRBOT 转换数据只有观测、状态和成功示范动作，没有逐步 reward、next observation、done/truncation 或失败轨迹，因此不能直接运行标准 offline RL。
2. 策略预测 `8×7` 的绝对关节目标，但部署每次只执行前 4 步；RL 的一个 decision step 必须明确为“重新观测后提交 4 步”，否则 Q 的时间尺度会和执行器不一致。
3. 真机在线探索既昂贵又有碰撞风险。先冻结视觉/语言特征、只训练 residual head 和 critic，再在仿真获得稳定提升，才有可审计的真机路径。

推荐的第一轮实验矩阵如下：

| 实验 | actor | critic/数据 | 目的 |
|---|---|---|---|
| A | BC checkpoint，完全冻结 | 无 | 固定成功率基线、记录动作约束 |
| B | IQL residual，先离线 | 示范 + 失败/扰动 transition | 检验仅靠已有数据能否提升 |
| C | TD3+BC residual，先离线 | 同上 | 与 IQL 的连续动作确定性基线比较 |
| D | residual SAC 或 TD3，仿真在线 | 离线 replay 与新 rollout 混合 | 用真实 success reward 改正 BC 的失败状态 |
| E | 解冻最后融合层/状态层 | 同 D | 测试视觉 backbone 是否需要适配；只有 E 有稳定收益时再扩大解冻范围 |

每个实验使用相同的固定初始状态、随机种子、最大步数和安全约束；主要指标是任务成功率、失败类型、约束触发次数、每次成功所需决策步数和推理延迟，动作 loss 只作为辅助指标。

## 当前策略和环境的事实边界

### AIRBOT 深度策略

- [airbot_depth/model.py](/home/user/wang-sm/depth-mani/airbot_depth/model.py) 的 `AirbotPolicyModel` 输入两个视角的深度/有效性通道、7D 状态和固定 prompt，输出 `horizon×7`；训练默认 `horizon=8`。
- [airbot_depth/train.py](/home/user/wang-sm/depth-mani/airbot_depth/train.py) 从随机初始化开始，用 AdamW 和 masked Smooth L1 拟合归一化的绝对关节位置；Depth Anything 只做冻结的 RGB→深度预处理，不在策略反向传播中更新。
- [airbot_depth/data.py](/home/user/wang-sm/depth-mani/airbot_depth/data.py) 只把每个 episode 的状态、动作和深度缓存成 BC 样本，并在 episode 尾部用 mask 忽略不存在的未来动作。它没有 RL transition 所需的 reward、next observation、终止类型或行为策略概率。
- [data/airbot-paperbag200-da2-20260924/manifest.json](/home/user/wang-sm/depth-mani/data/airbot-paperbag200-da2-20260924/manifest.json) 记录 200 条、38,923 帧、180/20 episode 划分；[data/airbot-can100-da2-20260929/manifest.json](/home/user/wang-sm/depth-mani/data/airbot-can100-da2-20260929/manifest.json) 记录 100 条、19,701 帧、90/10 划分。两份数据的 `action_semantics` 都是 `absolute_joint_position`。
- [docs/airbot-paperbag-depth.md](/home/user/wang-sm/depth-mani/docs/airbot-paperbag-depth.md) 记录部署时每次最多执行 4 个目标、策略输出长度为 8、数据频率 25 Hz，以及真实接入前必须核验关节限位和深度预处理一致性。

因此 AIRBOT 的 RL action 可以定义为绝对关节目标，但必须在环境 wrapper 中计算并限制相邻目标的速度、加速度和关节范围。不要把 LIBERO 的 OSC_POSE 增量动作直接当作 AIRBOT action。

### LIBERO/robosuite 仿真

- [depth_policy/simulation.py](/home/user/wang-sm/depth-mani/depth_policy/simulation.py) 创建 LIBERO `OffScreenRenderEnv`，使用 robosuite `OSC_POSE`、7D action、深度相机和固定控制频率。
- [configs/wine.json](/home/user/wang-sm/depth-mani/configs/wine.json) 和 [configs/cream_cheese.json](/home/user/wang-sm/depth-mani/configs/cream_cheese.json) 将控制频率设为 20 Hz、最大 300 步；[depth_policy/collect.py](/home/user/wang-sm/depth-mani/depth_policy/collect.py) 默认每次推理后取 action chunk 的前 4 或 5 步，再重新观察。
- LIBERO 的任务、初始状态和 success check 来自作者仓库 [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)；robosuite 的控制器、相机和仿真接口来自 [StanfordVL/robosuite](https://github.com/ARISE-Initiative/robosuite) 官方仓库。可扩展任务和 GPU 物理仿真时，也应先按 [ManiSkill 官方文档](https://maniskill.readthedocs.io/en/latest/) 固定 observation/action/reward 接口。
- [openpi 的训练配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) 将 `action_horizon` 定义为 action chunk 长度；当前工程自己的 AIRBOT policy 是独立的 CNN/MLP/GRU，而不是直接对 pi05 的 flow-matching 主干做 RL。两者应分别记录 checkpoint 和 action semantics。

## 方法比较

### 离线 RL：优先于真机探索

| 方法 | 原论文/作者实现 | 核心做法 | 对当前任务的适配 | 主要风险 |
|---|---|---|---|---|
| IQL | [Implicit Q-Learning](https://arxiv.org/abs/2110.06169)，[作者实现](https://github.com/ikostrikov/implicit_q_learning) | 不在数据集外显式最大化动作；用 expectile value 学习和 advantage-weighted actor 更新，适合固定数据集 | **首选 offline baseline**。示范动作分布窄时比直接 Q-learning 稳定；actor 可初始化为 BC 并限制在 residual 范围内 | 只有成功示范、没有失败/负回报时，Q 的排序信息不足；视觉输入需先固定 encoder 或用低维状态 sanity check |
| AWAC | [Advantage Weighted Actor Critic](https://arxiv.org/abs/2006.09359)，[rlkit 实现](https://github.com/rail-berkeley/rlkit) | 用 `exp(A/λ)` 对行为数据加权，策略更新仍接近 BC | 适合从 BC 平滑过渡；可先训练 critic，再对动作头做小步更新 | advantage 估计偏差会把极少数异常动作权重放大；需裁剪权重并保留 BC 正则 |
| CQL | [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)，[作者代码](https://github.com/aviralkumar2907/CQL) | 对数据集外动作压低 Q，缓解 offline extrapolation error | 当收集到大量失败和扰动动作、且需要严格保守时有用 | 视觉策略和 8×7 chunk 会使 Q 学习成本高；保守项过强会退化成 BC，过弱仍会产生 OOD action |
| TD3+BC | [TD3+BC](https://arxiv.org/abs/2106.06860)，[作者代码](https://github.com/sfujim/TD3_BC) | TD3 双 critic 加行为克隆正则和 Q 归一化 | **最容易做的确定性连续动作 baseline**；与当前确定性 BC head 形态接近 | 原方法按单步连续 action 给出；对 chunk 要明确一个 transition 是 4 步执行块，并在 actor 中对每个输出步加动作平滑/边界约束 |

当前数据若没有新增失败轨迹，IQL/AWAC/CQL/TD3+BC 都不能凭空知道“夹偏、碰撞、没抓住”比示范差多少。至少要收集成功和失败/扰动两类 transition，并记录每个 episode 的 success、timeout、碰撞和安全终止原因。

### 在线 RL：先仿真、后小规模真机

| 方法 | 原论文/官方资料 | 需要的 actor 形式 | 适配判断 |
|---|---|---|---|
| PPO | [原论文](https://arxiv.org/abs/1707.06347)，[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | 通常是带 log-prob 的随机 Gaussian actor，按 on-policy rollout 更新 clipped surrogate objective | 对高维视觉策略样本效率差，且当前确定性 actor 没有 log-prob；如采用 PPO，应只训练低维 residual head，并在仿真中使用 Gaussian residual |
| SAC | [原论文](https://arxiv.org/abs/1801.01290)，[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html) | 随机 actor、双 Q、熵正则和 replay buffer | **首选在线方法**（仿真或安全受限台架）。用 `a = clip(a_BC + α tanh(δ))`，只让 residual 探索；保留离线 BC 数据防止遗忘 |
| TD3 | [原论文](https://arxiv.org/abs/1802.09477)，[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html) | 确定性 actor、双 Q、target policy smoothing 和 delayed update | 若希望保留确定性控制，TD3 比 PPO 更贴合；但探索噪声必须在仿真或安全动作空间内注入，不能把未经裁剪噪声直接发给机械臂 |

如果仿真 rollout 预算足够，可把 [RLPD](https://arxiv.org/abs/2302.02948) 作为第二阶段的 offline-to-online 对照；其[作者实现](https://github.com/ikostrikov/rlpd)用离线 replay 和在线 replay 混合训练 SAC，并用 critic ensemble 稳定价值估计。它比 residual SAC 更复杂，且需要大量安全在线数据，因此不应作为第一版真机方案。若离线策略到在线策略的分布间隔很大，再考虑 [Cal-QL](https://arxiv.org/abs/2303.05479) 一类校准保守 Q 方法。

PPO/SAC/TD3 的论文算法都是连续控制的通用形式，并没有替当前图像策略解决 reward、chunk 和安全约束。视觉 encoder 不应一开始和 actor、critic 同速更新；先冻结并验证低维 state-only 或 frozen-feature 版本，能把“RL 算法问题”和“视觉分布漂移问题”分开。

### 示范增强、残差和不确定性

- **DAPG**：论文 [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087) 的核心是把示范策略梯度和 RL 策略梯度相加，先用示范获得可行行为，再逐步让 reward 梯度接管。它比从随机策略开始的 PPO 更接近当前场景；当前实现可把 DAPG 的示范项实现为 chunk-level BC/residual loss，并逐步降低权重。
- **Residual RL**：论文 [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201) 把已有控制器作为基准，RL 只学习修正量。当前 BC policy 就是基准控制器，推荐形式为 `a_RL = clip(a_BC + α·δ, limits)`；`α` 从小值开始，且应按关节/夹爪分别归一化。这样视觉 policy 的已有能力被保留，RL 主要修复接触、末端对齐和收尾动作。
- **SOAR/IL-SOAR**：这个缩写不是唯一标准术语。若指作者仓库 [stefanoviel/SOAR-IL](https://github.com/stefanoviel/SOAR-IL)，其 README 描述的是以多个 Q 网络估计不确定性、用 uncertainty bonus 做探索的 SOAR 版本，基础版本用一个 Q 网络。它适合作为 critic ensemble/探索模块的参考；没有在当前 AIRBOT 或 LIBERO 接口上验证，不能直接当成成熟的后训练 recipe。

残差方法的关键不是把 `δ` 放大，而是让 actor 的输出范围、关节速度、末端 workspace 和 gripper 开合都在环境 wrapper 中硬限制；否则 residual 会很快离开 BC 数据分布，critic 的高 Q 也可能只是 extrapolation error。

### 奖励学习、偏好和模仿奖励

- 如果没有可靠的几何成功判定，可以参考 [GAIL](https://arxiv.org/abs/1606.03476) 和 [AIRL](https://arxiv.org/abs/1710.11248) 从示范学习 discriminator/reward，但这并不自动解决 sparse success；视觉 discriminator 在示范量小、视角固定时容易学到背景和时间进度。
- 如果人能逐段比较两条轨迹，可参考 [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)：训练 preference model，再把其输出作为 reward。对当前夹取任务，首轮更建议人工标注“抓住/没抓住、碰撞、掉落、是否回到 reset”这些离散事件，并把标注作为校验 success detector，而不是直接用一个黑箱 preference reward 驱动真机。
- 奖励应优先使用环境真值事件：成功终止 `+1`，安全终止/碰撞负奖励，timeout 为 0 或小负值；距离、姿态、夹爪接触等 shaping 采用潜势差并单独记录，避免 shaping 项压过成功目标。LIBERO 的 task success check 和当前 episode writer 可作为事件来源，而不能只用动作 loss 充当 reward。

如果需要把碰撞、力或 workspace 违规作为显式成本，可参考 [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) 的 reward/cost 分离形式；它不能替代硬限位和急停，且需要 rollout 中可靠的 cost 标签。对连续、高维视觉 action，先在 wrapper 中硬约束，再把成本送入 critic，通常比直接依赖 CPO 的期望约束更容易审计。

## 8 步 action chunk 的 MDP 定义

这是实现成败的核心。假设每次重规划执行 `K=4` 个动作，模型仍输出 `H=8`：

1. 在时刻 `t` 观察 `o_t`，actor 输出 `A_t=(a_t^0,…,a_t^7)`。
2. 环境只执行 `a_t^0…a_t^{K-1}`，每个动作都经过同一限位/速度 wrapper；执行期间记录每帧 observation、action、sim time、碰撞和 success。
3. 得到第 `t+K` 个 observation 后形成一条 RL transition：`(o_t, A_t[:K], r_t:t+K, o_{t+K}, done)`。如果任务在块内提前成功或安全终止，`done` 必须截断，并保留实际执行长度。
4. critic 的 discount 使用决策块时间尺度：若单帧折扣为 `γ_frame`，可用 `γ_block = γ_frame^K`；或者把块内累计 reward 明确记录为 `R_t = Σ γ_frame^i r_{t+i}`。
5. actor loss 只对可执行的前 `K` 步计算；后 4 步是下一次重规划前的预测，可作为平滑/一致性辅助项，但不能让 critic 假设它们已经执行。

对 AIRBOT，`a` 是归一化后再反归一化的绝对关节目标；对 LIBERO，`a` 是 OSC_POSE 的增量输入。两者分别训练 replay 和 normalization，不能共用一个 Q 网络。若希望把整段 8 步视作一个原子动作，则 critic 输入维度变成 56，且 reward 延迟、OOD 风险和 Q 学习难度都会明显增加；第一版应使用 `K=4` 的执行块。

## 必须补齐的数据和环境接口

### Transition schema

为每个决策块写入独立的 RL 数据集（保留原始 HDF5，不覆盖 BC 数据）：

```text
episode_id, decision_index, timestamp
obs: RGB/depth, state, prompt, preprocessing_digest
action_chunk: [H, 7], executed_action: [K, 7], executed_length
next_obs: same fields at t+K
reward_sum, success, collision, safety_stop, timeout, done, truncation
policy_id, checkpoint_sha256, seed, simulator_xml_sha256
```

AIRBOT 真机采集必须额外记录控制器实际接收的目标、限位前后的 action、相机时间戳、CAN/控制器错误和人工中止原因；只记录网络输出不足以重建 transition。

### Reward and reset

1. 先实现可重复的 `reset(seed, initial_state)`、`step(action_chunk)`、`check_success()`、`check_collision()`、`check_limits()` 和 `termination_reason`。
2. 对 LIBERO 使用官方 task success；对 AIRBOT 先建立纸袋/易拉罐的任务判定（物体是否被夹爪保持、是否到 reset 区、是否掉落），再对少量轨迹逐帧人工复核。
3. 记录成功和失败的初始状态分布；不能只反复采样已有成功初始状态，否则 offline critic 学不到失败边界。
4. 先做 state-only 或冻结图像 feature 的 reward/critic sanity check，再启用 RGB→深度流水线。任何 Depth Anything 权重、预处理尺寸、相机顺序变化都应写进 `preprocessing_digest`。

### 数据收集顺序

1. 用当前 BC checkpoint 在固定 LIBERO 初始状态上跑完整评测，保存每个失败状态和第一处安全违规。
2. 在仿真中对 BC action 加小幅、已限幅的末端/关节扰动，并收集成功、失败、碰撞、超时四类轨迹；同时做相机、物体位姿、摩擦和延迟随机化。
3. 用这些 transition 训练 IQL、TD3+BC；离线评测只在完全隔离的初始状态和 episode 上进行。
4. 仅当离线策略不降低 BC 成功率，才开启 residual SAC/TD3 仿真在线 rollout；replay 中固定保留 BC/成功数据，防止策略遗忘。
5. 真机先做 shadow mode（只推理和记录，不下发 action），然后做无物体/软目标的低幅 residual，最后才做完整任务；每一步都允许人工接管。

## 推荐实施计划

### Phase 0：基线和接口验收

- 固定 BC checkpoint、环境 commit、模型/深度 digest、action normalization、`K=4/H=8` 和初始状态 bank。
- 用当前 [depth_policy/evaluate.py](/home/user/wang-sm/depth-mani/depth_policy/evaluate.py) 流程跑 baseline，并把每个 episode 记录扩展成 RL transition；先确认 success/timeout/exception 没有混淆。
- 验收标准是同一 seed 重放得到一致的 observation、action 和 termination；否则不要比较算法。

### Phase 1：离线 critic 和 residual actor

- 先用低维 state 或 frozen image features 训练 IQL 与 TD3+BC；actor 初始化为 BC，residual 输出零均值、小范围 `δ`。
- 每个 batch 同时采样 demonstration、成功 rollout、失败/扰动 rollout；保留 BC loss（例如 residual actor 的动作偏离惩罚）作为约束，并监测 Q ensemble disagreement。
- 离线选择标准不能是 Q 值最大，而是隔离初始状态上的真实成功率、碰撞率和动作边界触发率。

### Phase 2：仿真在线 fine-tuning

- 先比较 residual TD3 和 residual SAC；SAC 的熵项只用于仿真探索，真机阶段把探索噪声关掉或设为经过审核的极小值。
- critic replay 采用离线/在线混合，并对 success、collision、timeout 分层采样；在线策略每次更新后都跑固定评测 bank。
- 若 critic ensemble disagreement 超过阈值，回退到 BC action 或缩小 `α`；这比让不确定 Q 驱动更大动作更安全。

### Phase 3：真机小步验证

- 在控制器外层实现硬限位、速度/加速度限幅、workspace 检查、动作 watchdog、相机过期检测和急停；所有拒绝/截断都要写日志。
- 先下发 BC 和 residual 混合动作的 shadow/低风险版本，设置人工确认和随时接管；不直接在线运行 PPO/SAC 的自由探索。
- 每个新 checkpoint 必须通过离线复算、固定仿真 bank、无物体动作安全检查和短时真机 canary 后才允许扩大 rollout。

## 安全和审计要求

算法论文通常假定 action 可以自由执行；机器人部署必须另加控制层。可将 [ISO 10218-1:2025](https://www.iso.org/standard/73934.html) 和 [ISO/TS 15066](https://www.iso.org/standard/62996.html) 作为工业机器人/协作机器人风险控制的外部约束参考，但不能把符合代码接口等同于标准认证。

至少需要：

- action shield 在 RL actor 之后、机器人控制器之前执行，硬拒绝越过关节、速度、加速度、末端 workspace 和夹爪安全范围的命令；记录原始与裁剪后 action。
- 过期/缺失/非有限图像、状态或推理超时立即保持安全状态或回退 BC，不发送上一块未经重新验证的 action。
- 训练、评估和真机部署分离 checkpoint；记录 source identity、环境 XML、深度模型 provenance、随机种子和代码 SHA。
- 失败轨迹不可静默删除；`exception`、`interrupted`、`timeout` 和 `collision` 要分别计数。
- 用固定验证 bank 作为门禁，要求新策略同时满足成功率不下降、碰撞率不升高、约束触发率在阈值内；不要根据 offline Q 或动作 loss 单独放行。

## 参考资料（primary sources）

### 算法和示范/奖励

- [Implicit Q-Learning（IQL）](https://arxiv.org/abs/2110.06169)；[官方实现](https://github.com/ikostrikov/implicit_q_learning)
- [Advantage Weighted Actor Critic（AWAC）](https://arxiv.org/abs/2006.09359)；[rlkit 实现](https://github.com/rail-berkeley/rlkit)
- [Conservative Q-Learning（CQL）](https://arxiv.org/abs/2006.04779)；[官方实现](https://github.com/aviralkumar2907/CQL)
- [TD3+BC](https://arxiv.org/abs/2106.06860)；[官方实现](https://github.com/sfujim/TD3_BC)
- [PPO](https://arxiv.org/abs/1707.06347)；[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [SAC](https://arxiv.org/abs/1801.01290)；[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [TD3](https://arxiv.org/abs/1802.09477)；[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [RLPD](https://arxiv.org/abs/2302.02948)；[官方实现](https://github.com/ikostrikov/rlpd)
- [Cal-QL](https://arxiv.org/abs/2303.05479)
- [DAPG](https://arxiv.org/abs/1709.10087)
- [Residual RL](https://arxiv.org/abs/1812.03201)
- [GAIL](https://arxiv.org/abs/1606.03476)；[AIRL](https://arxiv.org/abs/1710.11248)
- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
- [IL-SOAR 官方仓库](https://github.com/stefanoviel/SOAR-IL)

### 当前工程和环境

- [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)；[LIBERO 论文](https://arxiv.org/abs/2306.03310)
- [robosuite 官方仓库](https://github.com/ARISE-Initiative/robosuite)；[robosuite 文档](https://robosuite.ai/docs/overview.html)
- [ManiSkill 官方文档](https://maniskill.readthedocs.io/en/latest/)
- [Physical Intelligence OpenPI 官方仓库](https://github.com/Physical-Intelligence/openpi)
- 仓库内当前实现：[AIRBOT 模型](/home/user/wang-sm/depth-mani/airbot_depth/model.py)、[训练入口](/home/user/wang-sm/depth-mani/airbot_depth/train.py)、[AIRBOT 数据集](/home/user/wang-sm/depth-mani/airbot_depth/data.py)、[LIBERO 仿真](/home/user/wang-sm/depth-mani/depth_policy/simulation.py)、[episode 记录](/home/user/wang-sm/depth-mani/depth_policy/episodes.py)。

除上述链接外，本文没有修改训练代码、环境或现有数据；下一步应先实现并验收 transition/reward/action-shield 接口，再选择 IQL/TD3+BC 的最小实验，而不是直接扩展网络或接入真机探索。
