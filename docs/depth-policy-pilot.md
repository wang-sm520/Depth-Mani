# 深度策略首轮实验方案

日期：2026-09-22。本文为实验设计；实际实现、验证证据和未完成项见 `../reports/progress.md`。

## 任务变更（2026-09-22，用户确认）

当前改为 `libero_goal/put_the_wine_bottle_on_top_of_the_cabinet`，使用
`configs/wine.json`。指令由 LIBERO task.language 提供，为
`put the wine bottle on top of the cabinet`。下文碗任务方案保留作为历史记录；
输入、教师、控制、数据规范、从零训练及对照要求不变。

旧任务后台自动训练流程及采集已安全停止，旧 `data/teacher` 不删除、不混入新数据。
新任务数据为 `data/wine/teacher`，初态划分为 `data/wine/splits.json`：
50 个唯一官方初态，35/7/8 分属训练/验证/测试。先采两条检查并回放，再推进批量实验。

位置变化来自官方预生成初态，而不是每条轨迹重新进行任意位置采样。
实测 50 个初态中的酒瓶自由关节初始 x/y 跨度约 2.96/2.95 cm；
详见 `reports/wine-initial-positions.json`。这些是稳定等待前的初始位置，
不代表整个桌面随机布局。超过 35 条训练示范会重复训练初态，独立场景数不增加。

新任务采集入口（每次使用不同 LABEL，避免覆盖日志）：

```bash
bash scripts/sim_cpu.sh -m depth_policy.manifest --config configs/wine.json
bash scripts/run_teacher_cpu_batch.sh 2 train wine-preview-0 '' configs/wine.json
.venv/bin/python -m depth_policy.validate --config configs/wine.json --data data/wine/teacher --report reports/wine-validation.json
```

`depth_policy.experiment --config configs/wine.json` 已适配酒瓶的数据目录、
manifest、训练及评估配置，输出隔离在 `runs/<task>/`；不得恢复旧碗任务 supervisor。
酒瓶 100 条端到端流程已实际完成，三组 GPU 学生训练及 CPU 闭环评估通过运行验证；
结果见 `reports/wine-first100-results.md`，不再仅是代码预检。
可用 `--pilot-service` 等待 systemd 试跑服务并检查退出状态；该模式要求服务单元
结束后仍可查询（例如启动时配置 RemainAfterExit=yes），服务丢失或异常时拒绝继续。
当前已运行的试跑服务不满足已验证的持久单元前提，需结束后审计再安排后续流程，
不要仅凭 summary 文件存在就跳过退出状态核查。

## 目标与范围

### 100/300/500 学习曲线

100 条首轮深度学生验证 7/7、测试 8/8，满足继续扩充的操作条件。
300/500 条保持同一官方任务与 35/7/8 初态划分，已有保留集不重新采样。
每个数据量、每种模态均从随机初始化训练 10000 步、seed 0、batch 64，
不从较小数据量 checkpoint 接着训练。输出目录包含任务名及数据量。
扩充前检查原 100/300 条训练子集文件指纹和验证集指纹仍一致，确保曲线使用嵌套数据。
更多轨迹只增加已有训练初态的教师随机动作样本，不增加独立初态数量。

`depth_policy.experiment --target 300` 和 `--target 500` 已支持。
300 条阶段已完成，结果见 `reports/wine300-results.md`；500 条扩充已于 2026-09-23
按用户要求暂停，`depth-mani-wine-500.service` 已停止，状态为 `runtime/wine500-state.json`。
不自动恢复采集；用户随后要求的 300 条策略官方 50 次与新随机 100 次评估已完成，
详见 `reports/wine300-evaluation150-results.md`。500 条结果尚未完成。
按实测 CPU 教师速度，100→300 约需额外 8.7 小时采集，
100→500 合计约需 17.3 小时，超时失败会延长。
预算报告 `reports/wine-learning-curve-preflight.json`，500 条 RGB 图像缓存估计
8.33 GiB（不含程序、临时张量、验证集等），启动时可用内存约 57.5 GiB。

GPU 训练（用户已授权）：本项目原 `.venv` 复用的 torch 2.7.1+cu126
在本机 RTX 5090 上实际报 no kernel image，不能用于 GPU 训练。
只读使用 `/home/user/miniconda3/envs/atec/bin/python` 的 torch 2.7.0+cu128，
不安装、不升级该共享环境。真实三种学生的 GPU 优化预检及恢复测试已通过。
自动流程可指定 `--training-python /home/user/miniconda3/envs/atec/bin/python
--training-device cuda`；只对训练阶段开放 GPU，教师采集和仿真评估仍用原 CPU 环境。
GPU checkpoint 的 CPU 仿真闭环兼容性已在酒瓶 100 条首轮三组学生评估中验证，
详见 `reports/wine-first100-results.md` 和 `reports/wine300-results.md`；500 条结果仍未完成。

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/miniconda3/envs/atec/bin/python -m depth_policy.train \
  --data data/wine/teacher --modality depth --device cuda --steps 10000 --limit 100 \
  --output runs/put_the_wine_bottle_on_top_of_the_cabinet/depth-100-seed0
```

上述正式训练命令须在成功训练示范及独立验证数据准备完毕后执行。
GPU 单元测试需设 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免加载该环境中
与项目无关的 pytest 插件；未修改共享插件或其依赖。

从零训练小型学生策略：双视角深度 + 机器人当前状态 + 语言 → 动作。
学生不使用 RGB、不加载视觉或语言预训练参数。教师使用官方 pi05_libero checkpoint，
并非 pi05_base。教师只负责生成示范，不与学生联合训练。

本轮只验证单任务闭环可行性，不证明开放语言理解、跨物体泛化或真实深度传感器鲁棒性。

## 任务

- suite：libero_goal。
- task：put_the_bowl_on_the_plate；按名字查任务，不硬编码整数 ID。
- 原始指令：Put the bowl on the plate。
- 目标：把 akita_black_bowl_1 放到 plate_1 上。
- 保留原始场景及物体，不先修改 BDDL；其他物体包括酒瓶与奶酪包装。
- 选择理由：指令不依赖颜色或包装文字，碗和盘可按几何类别区分；这是选题判断，实际深度可见性仍需渲染检查。
- 限制：原始物体位置区域很窄，成功不自动证明形状识别。增加 state-only 对照，后续再做布局扰动。

任务来源：https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl
教师来源：https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md

## 输入与控制接口

- 教师：agentview RGB + robot0_eye_in_hand RGB + state + 原始指令。
- 学生：同两相机深度 + state + 原始指令；RGB 只归档与回放。
- 深度采集建议 256×256，训练缩放为 128×128，保留米制深度与有效性信息。
- state 对齐 openpi LIBERO：末端位置 3 + 轴角姿态 3 + 两指关节位置 2，共 8 维；运行时断言。
- action：沿用环境 OSC_POSE 的 7 维控制命令（位置、姿态控制量与夹爪），不是 7 维绝对关节角。
- 控制频率按当前 LIBERO wrapper 为 20 Hz，启动时核实并写元数据；仿真时间与推理墙钟时间分开。
- 教师先沿用每 5 个控制步重规划；每一步都记录，不仅记录查询教师的帧。
- 学生初始预测 horizon=8 的动作序列，每次执行前 4 步再重规划；这些是实验起始超参数，可调整。
- 不引入物体真值位姿或分割标签作为学生输入。

本地核查来源：
/home/user/wang-sm/openpi/examples/libero/main.py
/home/user/wang-sm/openpi/third_party/libero/libero/libero/envs/env_wrapper.py

## 数据记录

建议每 episode 一个压缩 HDF5 文件，RGB 回放视频单独保存。

轨迹级：
- instruction、episode_id、success。
- task_name、seed、initial_state_id、完整初始仿真状态、split。
- teacher checkpoint 标识、代码版本、相机与控制器配置、control_freq、depth_units、图像方向约定。
- termination_reason，区分任务成功、超时与异常；不能把任意 done 当成功。

控制时刻数组（长度 T）：
- depth_agentview：[T,H,W]，float32 米制，压缩无损存储。
- depth_wrist：[T,H,W]，float32 米制，压缩无损存储。
- state：[T,8]，float32。
- executed_action：[T,7]，float32，保存实际提交 env.step 的命令。
- step_index：[T]；sim_timestamp：[T]，秒。
- RGB：同一观测时刻的双视角原始图像可归档；MP4 仅作调试回放。

时序：读取 obs_t → 保存 D_t/state_t 与将执行的 a_t → env.step(a_t) → obs_(t+1)。
末尾观测可单独保存，但不得错配最后一个动作。启动稳定等待帧与示范段分开标识。
未来动作标签必须来自连续实际执行动作，不跨 episode；尾部 padding 必须设置 loss mask。

深度不能以伪彩色或普通 8-bit 有损视频作为训练原始数据。robosuite 渲染深度默认为 [0,1]
归一化缓冲，采集器需用当前版本相机工具转换到米制，并核实数值和方向，不能直接当米数使用。
不得逐帧 min-max 归一化而丢失跨帧尺度；训练预处理配置固定并随 checkpoint 保存。

深度来源：https://robosuite.ai/docs/modules/renderers.html

## 小型学生基线

以下为首轮设计，实际实现以 `../depth_policy/model.py` 为准：

双视角深度 → 共享的小型随机初始化 CNN，保留空间布局特征
语言 → 随机词嵌入 + 小型 GRU
state → MLP
融合 → 小型动作序列预测头 → [8,7]

先使用行为克隆和 masked Smooth L1 动作损失，训练集统计归一化 state/action。
目标参数预算约 5–15M，以实际实现统计为准；无需复刻 pi05 架构或先实现扩散模型。
固定单条指令时语言分支可能被忽略，这是首轮范围限制，不声称已学会语言选择。

## 执行阶段与评估

1. 环境烟雾测试：双视角 RGB-D 渲染、8 维状态与 7 维动作、成功判据和初始状态恢复。
2. 教师试跑 20–50 个初始状态，统计所有尝试的成功率与失败类型，再决定是否批量采集。
3. 先采约 100 条成功训练示范；另留独立验证初态，不按帧随机拆分。
4. 训练小模型并闭环测试；若有效，再按 100/300/500 条成功轨迹扩充学习曲线。
5. 对照保持机器人状态和预算一致：深度+state+语言、RGB+state+语言、state+语言。
6. 报告成功数/总数、初态数量、训练种子和数据量；在同一批保留初态上比较教师与学生。

初态划分优先于采集：相同 initial_state_id 的全部重复轨迹进入同一 split。
不要循环重放有限官方初态，然后把重复轨迹计作新增独立场景。若需新增初态，先验证合法采样、
可复现恢复和教师适应性，再采集；明确区分原始任务分布与修改后的任务分布。
首轮成功轨迹用于 BC，失败轨迹独立保留用于诊断，不默认为正确动作监督。

## 下一步

独立采集器及运行入口见 `../README.md`，不修改已有机器人真机任务。
启动前核实本机依赖、checkpoint 和 GPU 占用；实际执行状态见 `../reports/progress.md`。
