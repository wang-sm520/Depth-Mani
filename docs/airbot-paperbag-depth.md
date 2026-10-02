# AIRBOT Play 红纸袋：RGB → Depth Anything → 策略

本实验使用用户指定的后期200条红纸袋示范。离线先生成深度缓存；部署时使用同一变换处理实时RGB，再与关节状态、固定prompt一起输入策略，输出原生绝对关节目标。

真机人工启动入口、Orin客户端安装及完整命令见 [AIRBOT真机部署](/home/user/wang-sm/depth-mani/docs/airbot-paperbag-real-robot.md)。

首阶段配置：[airbot_paperbag_da2_20260924.json](../configs/airbot_paperbag_da2_20260924.json)。用户随后要求从50,000继续至100,000 step，接续配置见 [airbot_paperbag_da2_100k_20260924.json](../configs/airbot_paperbag_da2_100k_20260924.json)。接续实时状态以 `runtime/airbot-paperbag200-da2-100k-20260924-state.json` 为准，本文不是完成证明。

## 数据与模型

| 项目 | 设置 |
| --- | --- |
| 原始数据 | `/home/user/wang-sm/pi0.5/openpi_v0.2.0/data/pick_paperbag_bd_200_20260912` |
| 示范与帧数 | 200条，38,923帧，25 Hz |
| 划分 | 180条训练、20条验证；按整条episode划分，seed=20260924 |
| 相机顺序 | head：1920×1080 RGB；wrist：848×480 RGB |
| 状态和动作顺序 | `joint1.pos`…`joint6.pos`, `eef.pos` |
| 单位 | 六关节rad，夹爪开度m |
| 原始动作 | 绝对位置目标，转换和推理均不重复做delta |
| prompt | `pick up the red paper bag and hold in the reset position` |
| Depth Anything | V2 Small HF，commit `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| 策略 | 沿用当前CNN＋状态MLP＋语言Embedding/GRU＋融合MLP，约667万参数 |
| 训练 | 首阶段随机初始化至50,000 step；随后恢复完整状态接续至100,000；每10,000 step留权重 |
| 优化 | AdamW，batch=64，lr=0.0003，seed=0 |
| 动作块 | 预测8步；部署初始执行4步后重新观测 |

单任务示范只支持上述训练prompt；这不是能理解任意新指令的预训练语言策略。

普通Depth Anything输出相对逆深度，近处数值较大，没有米制尺度。每帧、每视角独立按2%/98%分位归一化到[0,1]，保留原RGB长宽比后居中补边到128×128。策略每个视角有深度和有效掩码两个通道，真实远处的0仍有效，补边无效。输入模型前统一float16取整再恢复float32；磁盘缓存保留float32，便于逐值复算。

原始RGB、state、action及时间戳不改写。HDF5缓存保存原生数值、实际视频PTS、深度和来源身份。头部视频的5个分段和腕部的单文件分别按episode偏移读取；不能用全局帧号直接索引每个视频。

纯深度会丢掉红色线索。如果部署场景要求在同形状袋子中按颜色挑选，当前输入不足以保留这项判断依据。相对深度也可能存在帧间尺度变化；离线和在线使用同一确定性预处理可以避免实现差异，但不保证消除深度估计本身的误差。

## 环境与运行

本机使用 `/home/user/miniconda3/envs/atec/bin/python`（PyTorch 2.7.0+cu128），支持RTX5090。不替换现有仿真或OpenPI环境。

首阶段的转换、50k训练和审计由 `depth-mani-airbot-paperbag-da2-20260924.service` 完成。接续工作流由 `depth-mani-airbot-paperbag100k-20260924.service` 托管，顺序执行完整状态续训、父链审计和推理包导出；不连接机械臂。重复运行前先核查状态：

```bash
cd /home/user/wang-sm/depth-mani
systemctl --user status depth-mani-airbot-paperbag100k-20260924.service --no-pager
jq '{status,stage,completed_stages,log,error}' runtime/airbot-paperbag200-da2-100k-20260924-state.json
jq '{status,episodes:(.episodes|length),frames:(.episodes|map(.frames)|add)}' data/airbot-paperbag200-da2-20260924/manifest.json
```

如需对另一份已确认的AIRBOT数据运行转换，使用新的输出目录。以下是本实验转换入口；当前任务已运行时不要重复执行：

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/user/miniconda3/envs/atec/bin/python -m airbot_depth.convert \
  --source /home/user/wang-sm/pi0.5/openpi_v0.2.0/data/pick_paperbag_bd_200_20260912 \
  --output data/airbot-paperbag200-da2-20260924 \
  --device cuda --local-files-only --validation-fraction 0.1 --split-seed 20260924
```

转换支持显式`--resume`，要求原始数据、划分、代码和深度模型来源完全一致。原训练入口 `python -m airbot_depth.train --help` 用于从零训练，拒绝非空输出。完整状态接续使用独立入口 `python scripts/resume_airbot_depth.py --help`，从已审计父run的`latest.pt`恢复模型、AdamW、采样器及Python/NumPy/Torch/CUDA随机状态，并写入新目录。

接续保留原10k–50k权重与metrics的原始字节，新增60k–100k权重。`best.pt`始终按全历史验证损失选择，可能仍是父阶段权重；`best_continuation.pt`单独保存50k之后的最佳权重。恢复校验、首批采样、优化器步号及父run/checkpoint的SHA均记入新run和审计，不把旧最佳权重重新标成100k。

首阶段实测50,000 step训练循环477.53秒，约9.55ms/step。以35,085训练帧、batch64折算，一个等效epoch约548.2 step、5.24秒。采样为有放回随机抽样，不是逐帧恰好遍历一遍；耗时仅指缓存深度后的策略训练，包含定期验证和保存，不含原RGB转深度或环境准备。

接续至100,000 step已完成：新增50,000 step实测474.87秒（7分55秒），平均9.50ms/step，等效epoch约5.21秒；两阶段训练循环累计952.40秒。全历史最佳权重为76,000 step，验证loss为0.0264022，比首阶段最佳下降3.18%。详见 [100k结果](/home/user/wang-sm/depth-mani/reports/airbot-paperbag200-da2-100k-20260924/results.md) 与 [已验证的推理包](/home/user/wang-sm/depth-mani/deploy/airbot-paperbag200-da2-100k-20260924/README.md)。

## 推理接口

训练完成后，`best.pt`按独立验证损失选择；`step_010000.pt`至`step_100000.pt`保留固定训练进度。`latest.pt`是最后保存点。所有权重都携带prompt、相机顺序、统计和深度来源身份。

Python调用（没有硬件操作）：

```python
from airbot_depth.policy import AirbotDepthPolicy

policy = AirbotDepthPolicy(
    "/home/user/wang-sm/depth-mani/runs/airbot-paperbag200-da2-100k-20260924/best.pt",
    device="cuda", local_files_only=True,
)
actions = policy.infer_rgb(
    state,  # shape [7], 六关节rad + 夹爪m
    {
        "observation.images.head": head_rgb,   # uint8 HWC RGB
        "observation.images.wrist": wrist_rgb,
    },
    prompt=policy.prompt,
)
# actions.shape == (8, 7)，原生绝对关节目标；此函数不发送硬件命令。
```

也可用 `python -m airbot_depth.policy --checkpoint ... --observation ...npz --output ...json --device cuda --local-files-only` 做离线推理。NPZ含`state`和两个相机原生键，或含`state`与已经使用同一变换生成的`depth`。

Websocket服务兼容现有OpenPI的AIRBOT远程策略客户端：

```bash
cd /home/user/wang-sm/depth-mani
env CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/user/miniconda3/envs/atec/bin/python -m airbot_depth.serve \
  --checkpoint runs/airbot-paperbag200-da2-100k-20260924/best.pt \
  --device cuda --local-files-only --host 127.0.0.1 --port 8026
```

| 客户端输入键 | 含义 |
| --- | --- |
| `observation/state` | shape[7]原生状态 |
| `observation/base_0_rgb` | head的uint8 RGB |
| `observation/left_wrist_0_rgb` | wrist的uint8 RGB |
| `prompt` | 上述精确训练prompt |

回复的`actions`为shape[8,7]。服务通过metadata声明25Hz、horizon=8、单位、相机映射和checkpoint SHA。原AIRBOT客户端的默认执行块可能是30，使用本模型时必须显式设置`--chunk-size-execute 4`，不超过8；数据对应动作步频为25Hz。同步客户端还包含推理停顿，25Hz动作发布不等于25Hz完整闭环；以真实两相机端到端延迟为准。

服务默认只监听localhost，无认证。远程机械臂客户端可通过已有可信SSH连接建立端口转发。此服务不导入机器人SDK、不使能电机。

换部署环境时必须重新验证深度预处理一致性：当前检查会拒绝库版本、模型文件或实现指纹变化。RTX5090的时延不能直接当作Orin的时延。不得通过删除一致性检查来掩盖差异。

## 验收与实机边界

100k正式输出位于 `reports/airbot-paperbag200-da2-100k-20260924/`，包括完整状态恢复与父链审计、离线在线深度及动作复算、验证集动作MAE、保持当前关节状态的基线、训练曲线、RGB/深度样例和实际推理延迟；首阶段来源/时间对齐审计保留在 `reports/airbot-paperbag200-da2-50k-20260924/`。离线动作误差不能表述为抓取成功率。

当前数据有1109帧、49条episode的关节2目标高于既有控制器0.17rad上限，最大0.176051rad。离线数据保留原值；实机接入前必须核对实际关节零位和限位，不能静默裁剪示范或放宽控制器保护。实际CAN接口、相机连接、reset姿态和硬件权限尚不属于离线验收结果。

奶酪盒任务已按用户要求取消，记录见 [cream-cheese-cancelled-20260924.md](cream-cheese-cancelled-20260924.md)，不要重启其后台服务。
