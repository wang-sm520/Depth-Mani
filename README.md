# Depth Policy Pilot

从零训练“外部深度＋腕部深度＋机器人状态＋语言 → 动作”的小型行为克隆策略。
教师为官方 `pi05_libero`；当前任务为 `libero_goal/put_the_wine_bottle_on_top_of_the_cabinet`。
RGB 仅用于归档与 RGB 对照，主实验不读取 RGB。实现状态与实测结果见 `reports/progress.md`。

100/300 条三组学生训练与真实闭环评估已完成，结果见 `reports/wine-first100-results.md`
和 `reports/wine300-results.md`；500 条扩充已按用户要求暂停，尚无最终结果。
`depth-mani-wine-500.service` 已停止，状态为 `runtime/wine500-state.json`；未经用户要求不恢复采集。
下列命令是复现入口，不要在该服务运行时重复执行采集、训练或评估。
旧碗任务配置 `configs/pilot.json` 和数据 `data/teacher` 仅作历史保留，不混入酒瓶实验。

固定 300 条深度策略的附加测试已完成：官方 50 初态 49/50，新随机 100 初态 100/100。
新随机仍受原任务放置区域约束，不是全桌面随机化；官方 50 包含训练初态。
测试范围、完整结果、初态文件与复现命令见 `reports/wine300-evaluation150-results.md`。

进一步只把酒瓶位置扩大到 10cm × 10cm（每轴 ±5cm），同一模型结果为 69/100。
这属于扩大范围评估，不是官方基准；详见 `reports/wine300-10cm-results.md`。

同一批冻结初态上的 RGB-300 对比已完成：原范围 **99/100**、扩大范围 **61/100**，
对应深度 **100/100**、**69/100**。两者训练条件一致，均使用验证集选择的 best.pt；
训练时间、配对结果与复现命令见 `reports/wine300-rgb-comparison-results.md`。

## 环境

本机配置在 `configs/wine.json`，依赖已有 openpi 与其 Python 环境，不是独立可移植的发行包。
新增仿真依赖装在本项目 `.venv` 中，未修改上游代码或环境。

| 组件 | 路径 / 约定 |
| --- | --- |
| 仿真与 CPU 评估 Python | `.venv/bin/python` |
| GPU 学生训练 Python | `/home/user/miniconda3/envs/atec/bin/python`，PyTorch 2.7.0+cu128 |
| 教师 Python | `/home/user/wang-sm/openpi/.venv/bin/python` |
| 上游 openpi | `configs/wine.json` 的 `openpi_root` |
| 教师权重 | `configs/wine.json` 的 `teacher_checkpoint`，本机已有，不随项目分发 |
| 初态划分 | `data/wine/splits.json`，固定 seed，按初态哈希分组 |
| 控制 | OSC_POSE，20 Hz，8 维状态、7 维环境动作 |

本机环境使用 `.venv/lib/python3.11/site-packages/local_shared.pth` 只读复用 openpi
的 PyTorch/NumPy 等包；本地安装版本见 `scripts/requirements-sim.txt`。
项目 `.venv` 的 torch 2.7.1+cu126 在 RTX 5090 上不兼容 GPU kernel，不能用它训练 CUDA 学生；
GPU 训练只读复用表中的兼容环境，不升级共享环境。以下安装命令仅用于重建，不需在现有环境重复执行。
重建时需先准备兼容的 openpi 环境，再建立对应 `.pth`，并运行：

```bash
uv pip install --python .venv/bin/python --no-deps -r scripts/requirements-sim.txt
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests
```

## 1. 初态与采集验证

```bash
bash scripts/sim_cpu.sh -m depth_policy.manifest --config configs/wine.json
```

现有 50 个不同初态划分为训练 35、验证 7、测试 8。100 条示范会包含对训练初态的重复采集，
不等于 100 个不同场景；不改变原始 BDDL，不把测试初态拿来训练或计算统计。

GPU 使用必须先检查并处理占用冲突。纯 CPU 仿真使用已验证的 `scripts/sim_cpu.sh`，
其 OSMesa 本地解包步骤与 JIT 兼容性处理见 `reports/progress.md`。
GPU EGL 渲染尚未验证，以下统一使用 CPU 仿真，教师与学生网络可单独使用 GPU。

```bash
bash scripts/sim_cpu.sh -m depth_policy.collect --config configs/wine.json \
  --dummy --output data/wine-smoke-motion --attempts 1 --max-steps 5
.venv/bin/python -m depth_policy.validate --config configs/wine.json --data data/wine-smoke-motion \
  --report reports/wine-smoke-validation.json --preview reports/wine-smoke-preview.png
```

`--dummy` 只做环境与记录测试，上述输出为 `data/wine-smoke-motion`，不提供训练示范、不算教师成功率。
回放检查需要指定实际生成的 episode 文件：

```bash
EPISODE='data/wine-smoke-motion/<episode_id>.h5'
bash scripts/sim_cpu.sh -m depth_policy.replay "$EPISODE"
```

## 2. 教师与示范

正式学习曲线保持 CPU 教师，GPU 教师仅作独立 smoke，不混入正式数据。
以下 GPU 服务是独立部署选项，不与 CPU 批次同时启动。仅在确认 GPU 可使用后启动。
`0.60` 是 JAX 分配器预算约定，不是硬件隔离保证；
启动和首次推理时仍需监测实际显存。

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 \
  /home/user/wang-sm/openpi/.venv/bin/python scripts/serve_teacher.py
```

服务只监听 `127.0.0.1:8015`，采集器验证 checkpoint 身份，按 episode seed 确定每次推理的随机种子。
未获 GPU 共享许可时，可使用已验证的 CPU 路径。先用真实观测做限时探测，再以低优先级、
4 个教师 CPU 核和 4 个仿真 CPU 核运行受时间预算约束的批次：

```bash
bash scripts/run_teacher_cpu_batch.sh 20 train wine-cpu-pilot-new-label '' configs/wine.json
```

第三个参数是本次日志标签，必须唯一；启动前拒绝替换已有教师服务，批次结束只关闭自己启动的服务。
本机实际首条成功轨迹约 2 分多钟，CPU 批量采集比 GPU 慢，不能假定所有轨迹同样短。

自动流程入口如下（需要已完成 pilot；本机正在运行 500 条流程，不要重复执行）：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -u -m depth_policy.experiment \
  --config configs/wine.json --label wine-reproduction-new-label --target 100 --steps 10000 \
  --pilot-pid-file runtime/wine-pilot20.pid --pilot-report reports/wine-pilot20-0-summary.json \
  --training-python /home/user/miniconda3/envs/atec/bin/python --training-device cuda
```

该入口等待指定 pilot 退出并检查 20 个不同初态、无异常、成功率不低于 80% 的运行门槛；
随后按批次采集、校验、训练三个模型并在同一保留初态上闭环评估。80% 是采集运行门槛，不是论文结论。
资源不足或子命令失败会停止，日志和 PID 记录于 `runtime/<label>-state.json`。
最终仍需人工/代理审计真实结果与条件性的 300/500 条实验，不会自动声称完整目标已达成。

若选择手动操作，下列采集命令需要先单独启动教师服务；不要与自动流程并行：

```bash
bash scripts/sim_cpu.sh -m depth_policy.collect --config configs/wine.json --split train --attempts 20
.venv/bin/python -m depth_policy.validate --config configs/wine.json \
  --data data/wine/teacher --report reports/wine-manual-pilot.json
```

查看全部尝试、失败原因、深度可视化与至少一条完整动作回放，通过后再扩充：

```bash
bash scripts/sim_cpu.sh -m depth_policy.collect --config configs/wine.json \
  --split train --attempts 200 --target-successes 100
bash scripts/sim_cpu.sh -m depth_policy.collect --config configs/wine.json --split validation --attempts 7
bash scripts/sim_cpu.sh -m depth_policy.collect --config configs/wine.json --split test --attempts 8
.venv/bin/python -m depth_policy.validate --config configs/wine.json \
  --data data/wine/teacher --report reports/wine-manual-data.json
.venv/bin/python -m depth_policy.summarize --data data/wine/teacher --output reports/wine-manual-summary.json
```

`--attempts` 是本次最多新增的尝试次数，`--target-successes` 是该 split 的累计成功目标。
不足目标不会伪报完成，需要查看报告再决定下一批。SIGINT/SIGTERM 在控制步边界保存后停止；
硬中断留下 `.partial.h5`，加载器忽略它。每条轨迹独立 UUID，不覆盖原文件。

## 3. 学生与对照

GPU 训练和 CPU 闭环评估已在 100/300 条阶段验证。主实验与两个对照使用相同数据与训练步数。
以下使用独立复现目录，不覆盖已有结果；数据必须先准备齐全，训练集不足时不应报告目标数据量完成。

```bash
for MODALITY in depth rgb state; do
  CUDA_VISIBLE_DEVICES=0 /home/user/miniconda3/envs/atec/bin/python -m depth_policy.train \
    --data data/wine/teacher --modality "$MODALITY" --device cuda --steps 10000 --limit 100 --seed 0 \
    --output "runs/wine-reproduction/${MODALITY}-100-seed0"
done
```

默认使用 CPU；仅显式 `--device cuda` 才进行 GPU 训练。三个模型均随机初始化，分别约
6.67M、6.67M、5.89M 参数。训练仅使用成功 train 示范，验证仅用于选择 `best.pt`。
`latest.pt` 包含优化器与随机数状态，使用相同命令追加 `--resume` 可续训；数据变更会拒绝续训。
训练输入图像为 128×128，深度固定裁剪到 3 米并附有效性通道，不逐帧拉伸。

## 4. 闭环评估

先用 validation 排查部署问题，再冻结模型选择与配置，在同一组 test 初态上比较。

```bash
for MODALITY in depth rgb state; do
  for SPLIT in validation test; do
    bash scripts/sim_cpu.sh -m depth_policy.evaluate --config configs/wine.json \
      --checkpoint "runs/wine-reproduction/${MODALITY}-100-seed0/best.pt" --device cpu \
      --split "$SPLIT" --repeats 1 --output "runs/wine-reproduction/eval-${MODALITY}-100-seed0-${SPLIT}"
  done
done
```

每次预测 8 步动作、执行前 4 步再重规划。结果与完整轨迹保存在对应评估目录。
报告成功数/尝试数、独立初态数、数据量、训练 seed、推理耗时；8 个 test 初态的结果精度有限，
重复执行同一初态不能包装成新增独立场景。根据首轮结果决定是否扩充 300/500 条，先核算磁盘。

## 数据格式

每 episode 一个无损压缩 HDF5。轨迹元数据含任务、指令、初态、seed、split、教师与代码身份、
相机与控制器配置。`steps/` 内含双视角 float32 米制深度、uint8 RGB、状态、实际执行动作、
step index、仿真时间、完整仿真状态与相机位姿。`phase=0` 是稳定等待段，`phase=1` 才是训练段。
保存的是动作执行前的观测；只有 `env.step` 返回后才将该观测与动作写入记录。
终止属性区分成功、超时、异常和中断；失败示范不参与 BC。

详细实验边界见 `docs/depth-policy-pilot.md`。目前任何单任务结果都不等价于语言泛化、
跨任务泛化或真实深度传感器鲁棒性；必须用真实闭环结果而非单元测试证明可行性。
