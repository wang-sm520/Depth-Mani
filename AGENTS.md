# AGENTS.md — AIRBOT 深度策略 RL 后训练（air-5090: ~/wang-sm/depth-mani）

本文件约束代码风格和工作方式，所有改动都要遵守。任务目标和进度见 `airbot_rl/TASK.md`。

## 1. 决策权

1. 需求、方案、仿真场景细节、算法和超参的取舍，只要不明确，先问用户。TASK.md 里已经定好的事项照做，不要重新讨论。
2. 安全门限和判据由用户来定：关节、速度、单步变化限位，工作空间禁区，残差幅度，夹爪范围，超时，成功判据，奖励数值，自动停止条件，真机准入条件。增删改数值或触发条件前，先向用户说明依据和备选做法，同意后才能改。不要换个名目变相加门限，比如在 env 里多加一个"保险"判断，或者在评估脚本里偷偷过滤。
3. 实验结果不理想时，按 TASK.md 里写好的应对方式处理。超出范围的，停下来汇报，不要自己改方案。
4. 用户一旦让你上真机测试，就默认环境已确认安全，可以直接调用真机，不用逐次确认。本任务目前只做仿真，不碰真机。

## 2. 代码风格（硬性要求）

以前的代码出过这些问题：3845 行里真正干活的不到三分之一；一个 70 行的训练入口写成了 350 行；10 个一次性脚本混在正式代码里。下面的规则就是针对这些问题的。

**写什么**
- 一个文件只做一件事。新功能先看现有模块能不能加一个函数解决，不行再新建文件。
- 直接用官方实现：Isaac Lab、rsl_rl、`airbot_depth` 里已有的类和函数（`AirbotDataset`、`batch_loss`、`evaluate_loss`、`save_checkpoint`、`DepthAnythingTransform`、`OnPolicyRunner` 等），不要复制改写。要改它们的行为，就用参数、hook 或子类。
- 数据流写成函数：输入什么张量，输出什么张量，一行 docstring 写清形状和单位。
- 常量放在定义它的模块里，只定义一次，别处 import。不要在两个文件里各写一份同样的数值。
- 能用张量批量算的，不要写 Python 循环逐个环境处理。

**不写什么**
- 不写防御性校验：sha256、文件锁、断点续跑、`.partial` 原子写入、逐字节比对、manifest 一致性检查、参数合法性检查、"重复确认"的 assert。数据是自己生成的，出错就让它直接报错。只在真正的外部边界（用户输入的路径、外部文件格式）做最少的检查。
- 不写元数据记录：`run.json`、`identity`、`provenance`、环境指纹、源码哈希。一个 `args.json` 加一个 `metrics.jsonl` 就够了。
- 不为假设的需求写代码：别名（`foo = bar`）、只用一次的 helper、`--smoke` 这类专用模式、可配置但没人改的选项、向后兼容的 shim。
- 不写长 docstring 和解释"做了什么"的注释。只在原因不明显时写一行注释：隐藏的约束、实测出来的数值、绕开 Isaac 某个 bug 的写法。
- 不写 `from __future__`、类型注解堆砌、`if __package__ in (None, "")` 这类模板代码。

**规模参考**：`train.py` 约 70 行，`finetune_bc.py` 约 80 行，`replay.py` 约 130 行，`eval_bc.py` 约 75 行。新脚本超过 150 行，先想想是不是写多了。

**一次性脚本**：诊断、对比、性能测试这类脚本放 `/tmp/rltest/`，不要提交进 `airbot_rl/`。结论写进汇报和 `TASK.md` 的进度记录里。

## 3. 工作方式

- 先读后写：动一个模块前，完整读一遍它和它调用的东西。
- 先小规模跑通，再放大：4 个环境、几个 step 跑通了，再上 128 个环境；3 条 episode 跑通了，再上全量。
- 每个里程碑完成后，在 `airbot_rl/TASK.md` 的"进度记录"里追加几行：做了什么、结果数值、产出路径、和预期不一样的地方。
- 评估时每个环境只统计它的第一个 episode（环境数 = episode 数）。如果一个环境跑多个 episode、凑够数就停，结束得早的 episode（多数是成功的）会被重复计入，成功率会偏高。
- 汇报只写事实和数字。失败就贴报错输出，跳过的步骤要说明原因。不要说"应该没问题"。
- 回复用中文，说人话，别用模板腔。

## 4. 本机约束

- GPU 训练和仿真统一用 `/home/user/miniconda3/envs/atec/bin/python`（Isaac Sim 5.1、Isaac Lab 0.54.4，源码在 `~/wang-sm/IsaacLab`；rsl_rl 5.0.1、torch 2.7.0+cu128）。从项目根目录运行，加 `PYTHONPATH=.`。不要升级或改动这个共享环境，也不要用项目的 `.venv` 跑 CUDA（它的 torch 不支持 5090）。
- 用 GPU 前先跑 `nvidia-smi`，看有没有别人在用。显存 32GB：128 个环境约占 29GB，同一时间只能跑一个 Isaac 任务。
- Isaac 进程出异常后可能卡住不退出、一直占着显存。清理时用 `pgrep -af` 找到 PID 再 `kill -9`；不要用 `pkill -f`，它可能把你自己的 shell 也杀掉。
- 跑超过 10 分钟的任务，用 `nohup ... > log 2>&1 &`，结束时写一个 done 文件，用轮询判断是否完成，不要阻塞等待。
- 不修改这些已有内容：`airbot_depth/`、`depth_policy/`、`runs/`、`data/`、`deploy/`、`airbot_deploy/`、`~/locomotion_zh/`（只读参考）。新代码全部放在 `airbot_rl/`。
- 磁盘剩余约 200G。每 2000 步存一个 BC checkpoint 约 60MB，PPO checkpoint 很小，不用担心。

## 5. 已知的坑

- MJCF 导入前要 `enable_extension("isaacsim.asset.importer.mjcf")`，`scene.py` 里已经做了。
- `PinholeCameraCfg.from_intrinsic_matrix` 必须显式传 `focal_length`，否则视场角会退化成约 89°。Omniverse 相机不支持主点偏移，所以先渲染居中的画布，再 `grid_sample` 到真实像素网格（`scene._pinhole_canvas`）。
- `render_interval` 不能设太大（1e5 会让 reset 卡死），1000 没问题。相机在决策点手动调用 `sim.render()`。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 必须在 CUDA 初始化前设好，`airbot_rl/__init__.py` 里已经处理。
- `@configclass` 的字段要实例化后才能访问（`CanEnvCfg().checkpoint`）。
- 每个 `ContactSensorCfg` 的 prim 只能对应一个刚体，所以每个臂链接各建一个传感器。
- DA2 一次推太多张会爆显存，`policy.head_depth` 按每批 32 张推理。
- 静态碰撞体（没有刚体的 cuboid）不支持 GPU 接触过滤，过滤后的接触力恒为 0。需要检测接触的物体必须是刚体（静止的用 kinematic）。
- 到 air-5090 的网络有时走中转，延迟高。远程长任务一律 nohup 加 done 文件。
- Isaac 进程偶尔写完输出后挂住不退出，要按 PID kill。
