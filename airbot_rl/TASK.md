# TASK.md — 腕部深度版 BC → 残差 PPO（Codex 长任务，M0–M4）

先读 `AGENTS.md`。本文件里写明的决定都已经由用户确认，照做即可，不要重新讨论。没写到的取舍先问用户。

## 0. 目标

AIRBOT Play 装在 GENISOM M1 狗背上（狗趴着），抓地上的易拉罐（直径 6.5cm，高 11.4cm，重 250g），抓起后回到初始位置。现有 BC 已在真机上验证，成功率约 30%。本任务在 Isaac Sim 里做残差 PPO 后训练：BC 冻结，PPO 输出一个有界残差，加到 BC 的动作上。

BC 的输入（用户已定，方案 A）：
- 头部：Depth Anything V2 相对深度。头部相机没有深度传感器，部署端照旧。
- 腕部：D405 米制深度，编码在 `airbot_rl/depth.py`（0.07–1.0m 线性映射到 [0,1]，越近越亮，超出量程或空洞记为无效）。部署端能拿到 D405 深度流；录制数据里没有 D405 深度，所以腕部只能靠仿真重放来训练。暂时不采真机深度。
- 状态：7 维（6 个关节 + 夹爪宽度）。25Hz，每次输出 8 步 chunk，执行前 4 步。

## 1. 已经完成的工作（不要重做）

- 场景（`scene.py`）：狗的趴姿和高度（BASE_Z=0.147，按头部深度误差扫描确定）；头部相机内参、畸变、曝光增益（0.65）；腕部相机外参（按 home 位姿下指尖的像素位置拟合）；用真实录像矫正出的地毯贴图；7m 外的墙；env_spacing 20m。
- 渲染：头部 1/4 分辨率 RGB（DA2 实际输入是 518×294，降分辨率后 BC 输出差 0.0016，小于相邻帧之间的 0.004）；腕部只渲染深度。
- 驱动（已用录制数据辨识，每个 action 保持 40ms）：肩 1000/40，腕 80/5，夹爪 1000/20，关重力（模拟真机重力补偿），命令延迟 100ms（`env.COMMAND_DELAY`）。验证集关节 MAE 0.0039 rad。
- 罐子起始位置（`assets/can_positions_fk.json`，97 条）：取夹持帧时两指中点的正运动学位置。夹持帧：从最大开度往后，第一个已经合拢 ≥1cm、且后两帧宽度变化都 <1mm 的帧。episode 10、65、76 已排除。
- 重放数据 `airbot_rl/datasets/replay_wrist_v1/`：`mixed/`（真实头部 DA2 + 仿真腕部深度）和 `sim/`（全仿真），各 97 条，87 条训练 / 10 条验证，动作都是真实动作。夹持帧之后罐子跟着 link6 走。
- 代码已精简到约 1150 行。旧版备份在 `airbot_rl/.backup-codex-20261002/`，其中 `identify_drive.py` 可以参考里面开环回放的写法。
- 实测吞吐量：128 个环境约 43 次决策/s，显存约 29GB。

## 2. 用户已确认的数值（不要改）

| 项 | 值 |
|---|---|
| 超时 `max_targets` | 1000 个 target（250 次决策） |
| 单步关节限幅 `max_joint_step` | 0.1 rad |
| 成功判据 | 夹爪中点回到初始位置 5cm 以内（`success_radius=0.05`），并且罐子最低点离地 > 0.25m（`success_can_bottom`）；满足后立刻结束 episode |
| 残差幅度 `residual_scale` | 关节 0.05 rad，夹爪 0.01 m |
| 奖励 | `success_reward=10`，`limit_penalty=100`/rad，`contact_penalty=1`/次决策（臂碰到狗身或地面，接触力 >1N），`topple_penalty=0.5`（每个 episode 一次，倾角 >45°） |
| PPO | rsl_rl PPO，`train.py` 里的 `TRAIN_CFG` 不改（init_std 0.5，lr 3e-4，adaptive KL 0.01，32 步/环境） |

把 `env.py` 里这几项的注释从 `proposed` 改成 `user-confirmed 10-03`，数值一个都不动。

## 3. 要注意的事

1. **原 checkpoint 不能接现在的 env**：它的腕部输入是 DA2，而 env 现在给的是米制深度。M0 冒烟测试可以用它，只是为了跑通流程，结果没有意义。M1 训练出新 BC 后，要把 `CanEnvCfg.checkpoint` 改成新的。
2. **接触检测的 bug**：地面是静态碰撞体，PhysX 不支持对它做 GPU 接触过滤，过滤后的力恒为 0（实测手指压地面时合力有 12.8N 和 24.6N，过滤后仍是 0）。所以之前所有"接触率"只统计了碰狗身。M0 要修掉。
3. **评估偏差**：老的 `eval_bc.py` 会让一个环境连续跑多个 episode，凑够数就停，短 episode（多数是成功的）会被重复计入，之前的成功率（12%、26%）可能偏高。M0 要改成每个环境只统计第一个 episode。
4. **不要重新生成重放数据**：DA2 改成 GPU 预处理后，结果和现在的 CPU 版本只差 0.0008，不影响训练。
5. 头部 DA2 的版本校验（provenance）不能去掉，它保证仿真和部署用的是同一个 DA2。
6. 进 PPO 的门槛、奖励、判据都是用户定的。结果不好时，只能按第 6 节允许的方式自己再试一次。

## 4. M0：基础设施修复和提速（所有评估都在 M0 之后进行）

先做快照：`cp -r airbot_rl /tmp/rltest/snapshot_m0`。然后按顺序做，每一项都要验收：

**M0.1 修复地面接触检测**
- 把 `scene.py` 里的 floor 改成 kinematic 刚体（`rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)`，保持不可见）。
- 验收：在 `/tmp/rltest/` 写一个脚本，命令臂压向地面（例如 j2=-2.9、j3=2.2，罐子先移开），检查左右手指传感器在地面那一列的过滤后接触力 >1N。启动日志里不能再出现 "GPU contact filter for collider ... Floor is not supported"。另外确认 home 姿态下各链路的接触力为 0，也就是没有误报。

**M0.2 DA2 改为 GPU 预处理 + fp16（只用于仿真，部署端不变）**
- 在 `policy.head_depth` 里，把 HF 的 CPU processor 换成 torch 实现：先用 processor 对一张图算出目标尺寸并缓存；然后 GPU 上做 bicubic 缩放（align_corners=False）、clamp 到 0–255、round、除以 255，再按 ImageNet 均值方差归一化；模型在 `torch.autocast(fp16)` 下推理。percentile 归一化继续用 `normalize_relative_depth`。
- 验收（我已经实测过一次，你要复现）：60 张真实头部画面（1/4 分辨率）上，和 CPU 版本相比，头部深度平均差 ≤0.001，BC chunk 平均差 ≤0.0005；128 张图耗时约 460ms（原来约 1100ms）。达不到就保留 CPU 版本，并汇报原因。

**M0.3 物理步长从 5ms 改为 10ms**
- 用户问过：这不影响部署频率。部署端仍是 25Hz、每个 waypoint 40ms；物理步长只是仿真内部的积分精度。改法：`PHYSICS_HZ=100`，`SUBSTEPS=4`，`COMMAND_DELAY=10`（仍然是 100ms）。
- 验收 A（驱动）：开环回放 10 条验证集 episode 的真实动作，每个 action 保持 40ms，带延迟，不放罐子。关节 MAE 和 5ms 时（约 0.0039 rad）相比不能高出 10% 以上。
- 验收 B（抓取）：挑 30 条 episode，带物理开环回放真实动作，罐子放在 FK 起始位置，按成功判据统计成功率。5ms 和 10ms 各跑一遍，成功率差 ≤5 个百分点。这个"真实动作开环回放的成功率"本身也要汇报，它反映的是仿真抓取的保真度上限。
- 任一项不达标，就退回 5ms，并汇报原因。

**M0.4 减少读取接触的开销**
- 查看 `~/wang-sm/IsaacLab` 里 ContactSensorData 有没有过滤后接触力的历史（`force_matrix_w_history` 之类）。如果有：设 `history_length=SUBSTEPS`，每个 waypoint 读一次，取这段时间内的最大值；不要每个物理步都读。
- 验收：64 个环境、20 次决策里，逐步读和按历史读得到的 `touched` 每一步都完全一致。没有这个历史字段，就跳过本项，并说明原因。

**M0.5 修正评估偏差**
- `eval_bc.py`：环境数等于 episode 数，每个环境只统计第一个 episode，全部结束就停。超过 128 个 episode 时，分多次启动，用不同 seed（seed 0、1……）。
- 在 summary 里加接近高度指标：每个 episode 里，取罐子还没离地（罐子最低点 < 0.01m）、并且夹爪中点到罐轴水平距离在 4–7cm 之间的那些决策，记下夹爪中点高度的最小值；对所有进入过这个区间的 episode 取中位数，报告为 `approach_height_median`；同时报告从没进入过这个区间的 episode 比例。真实示教的中位数是 24.5cm。
- 加 `--ppo <model_N.pt>` 参数：加载 rsl_rl actor，用确定性均值作为动作。不加这个参数时，残差为 0，即纯 BC。

**M0.6 冒烟测试**
- 用原 checkpoint，128 个环境跑 `train.py` 3 个 iteration，确认能跑通；接着用 `--resume` 再跑 1 个，确认能恢复。报告每个 iteration 的耗时，和改动前的约 94 秒对比。
- 在 env 里加 episode 级别的成功率日志：统计这一步结束的 episode 里成功了几个，写进 `extras["log"]["episode_success"]`，供 M3 判断是否提前停止。

M0 结束后，在 TASK.md 的进度记录里写下：每项验收的数值，最终采用的物理步长，每个 iteration 的耗时。

## 5. M1：训练腕部深度版 BC

```
python airbot_rl/finetune_bc.py --checkpoint runs/airbot-can100-da2-50k-20260929/best.pt \
  --data airbot_rl/datasets/replay_wrist_v1/mixed airbot_rl/datasets/replay_wrist_v1/sim --weights 0.5 0.5 \
  --output airbot_rl/runs/bc_wrist_v1 --steps 20000 --lr 1e-4 --batch 32 --eval-every 2000
```
（用户已确认：从原 checkpoint 初始化，全参数训练，mixed 和 sim 各占 0.5，lr 1e-4，20000 步，每域每批 32 帧。）

- 选模型：step 2000、4000、……、20000 共 10 个 checkpoint，各用 `eval_bc.py` 跑 50 个 episode（seed 0）。按成功率排序；成功率相同时，看 `approach_height_median` 更接近 24.5cm 的。验证 loss 只作参考，不用来选模型。
- 选出的 checkpoint 再用 seed 1 跑 50 次，报告两次合计 100 次的结果。
- 用 `record_video.py` 给选中的 checkpoint 录 4 个环境的视频。
- 把 `CanEnvCfg.checkpoint` 改成选中的 checkpoint。
- 报告内容：每个 checkpoint 的成功率、侧翻率、接触率（地面和狗身分开统计）、`approach_height_median`、mixed 和 sim 的验证 loss；失败方式分类（成功、侧翻、悬停不下降、没有接近罐子、碰撞）。

## 6. M2：进 PPO 的门槛

**门槛（用户定）**：选中的 BC 在 100 次评估里，成功率 ≥ 10%，并且 `approach_height_median` ≥ 15cm。两条都满足才能进 M3。

**没过时**：允许自己再试一次，只能从下面三种改法里选一种，并说明为什么选它：
- (a) 数据权重改成 mixed 0.3 / sim 0.7（如果主要问题是仿真头部画面下表现差）；
- (b) 步数改成 40000（如果验证 loss 还在下降）；
- (c) lr 改成 3e-5（如果验证 loss 很早就开始回升，说明过拟合）。

用和 M1 一样的流程训练、评估。还是没过，就停下来交报告：两次的评估表、视频、失败方式分类，以及你认为的原因。门槛、判据、奖励都不能改。

## 7. M3：PPO

- 试跑：128 个环境，30 个 iteration。以下情况属于异常，出现就停下来汇报：loss 或动作出现 NaN/inf；value loss 持续增长；episode 成功率连续 10 个 iteration 低于 BC 成功率的一半；超过一半的残差维度饱和（tanh 后的绝对值 >0.95）。
- 正式训练：不出现异常，就接着试跑的结果 `--resume`，跑到最多 1000 个 iteration。如果 `episode_success` 的 20 个 iteration 滑动平均连续 200 个 iteration 都没有超过之前的最好值，就提前停止。每 25 个 iteration 存一次 checkpoint。
- 长任务用 nohup，每 100 个 iteration 往进度记录里追加一行：iteration、滑动成功率、平均奖励、残差幅度。

## 8. M4：最终评估

- 按训练日志里的 episode 成功率，挑出最好的 3 个 PPO checkpoint，各用 `eval_bc.py --ppo` 跑 100 次（seed 0），选出最好的一个。
- 最终对比：选中的 PPO 和 M1 选出的 BC，各跑 200 次（seed 0 和 seed 1 各 100 次），用同样的 seed。
- 报告：成功率、侧翻率、接触率（地面和狗身分开）、`approach_height_median`、平均越限量、残差幅度分布、失败方式分类，PPO 和 BC 并排对比。各录 4 个环境的视频。
- 最后在 TASK.md 写总结：产出路径、关键数字、仿真和真机之间还剩哪些已知差距（腕部从没见过真实 D405 噪声；仿真里 RGB 头部画面和真实画面的差异等）。本轮到此结束，不做部署端的改动。

## 9. 进度记录

（每个里程碑追加在这里）

- 2026-10-03 M0.1：home 姿态下过滤接触力最大值 `0.000 N`；压地回放时左/右手指分别为 `17.618 N`/`25.217 N`，均超过 1 N；启动日志无旧的 Floor GPU contact filter 警告。日志：`/tmp/rltest/m0_contact_floor.log`。
- 2026-10-03 M0.2：60 张真实头部画面深度均值差 `0.0001713`，BC chunk 均值差 `0.0001501`；128 张 GPU 预处理约 `445.6 ms`，验收通过。
- 2026-10-03 M0.4：64 环境、20 次决策逐物理步与历史接触读取的 `touched` 逐步比较总不一致 `0`，验收通过。日志：`/tmp/rltest/m0_contact_history.log`。
- 2026-10-03 M0.3：10 条验证集开环驱动回放在 5 ms/10 ms 下的关节 MAE 分别为 `0.0038764053`/`0.0038699764` rad，比值 `0.99834`，驱动验收通过。相同的 30 条 FK 罐子起始位置开环抓取，5 ms 成功 `13/30`（`43.33%`），10 ms 成功 `12/30`（`40.00%`），差 `3.33` 个百分点，抓取验收通过。最终采用 10 ms 物理步长（100 Hz，4 substeps，100 ms 延迟对应 10 个物理步）。结果：`/tmp/rltest/m03_step_compare.json`；分批日志：`/tmp/rltest/m03_200_b*.log`、`/tmp/rltest/m03_100_b*.log`。
- 2026-10-03 M0.5：`eval_bc.py` 已静态检查通过，并用 4 环境跑通首个 episode 统计；每个 episode 均计 250 次决策，输出 `approach_height_median` 与 `approach_no_entry_rate` 字段。冒烟结果成功率 `0`、接触率 `0`、侧翻率 `0`、未进入接近区比例 `1.0`；结果：`/tmp/rltest/m05_eval_smoke/summary.json`。
- 2026-10-03 M0.6：原 BC checkpoint、128 环境初始 smoke 的 3 个 iteration 耗时分别为 `51.04 s`、`51.14 s`、`50.63 s`，均无 NaN/inf；从最新保存的 `model_2.pt` 恢复 1 个 iteration 成功，恢复日志 iteration `2/3`、耗时 `53.78 s`。日志和状态：`/tmp/rltest/m06_smoke/initial.log`、`/tmp/rltest/m06_smoke/resume.log`、`/tmp/rltest/m06_smoke.done`。
