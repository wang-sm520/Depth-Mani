# AIRBOT 红纸袋：人工启动深度策略

5090运行RGB→Depth Anything V2→策略；接有机械臂和相机的Orin运行采集与动作客户端。使用200条、25 Hz数据训练至100,000 step后选出的76,000 step最佳权重。策略服务在前台运行，机械臂由操作者显式执行命令并按Enter启动。

本文件说明分机部署的客户端。Depth Anything和策略均在Orin本机运行时，使用独立Orin部署包及其顶层README。

| 项目 | 固定设置 |
| --- | --- |
| 策略输入 | head原图1920×1080、wrist原图848×480，均uint8 RGB；原生7维状态；训练prompt |
| prompt | `pick up the red paper bag and hold in the reset position` |
| 策略输出 | 8×7绝对位置目标：六关节rad、夹爪m；每次最多执行4个完整目标再观测 |
| 动作发布 | 25 Hz；保留0.01 rad / 0.005 m步长插值，推理和插值会增加实际执行时间 |
| 底层控制 | profile显式固定100 Hz；发送延迟后等待完整周期，不补发错过的周期 |
| 模型环境 | 5090已有`/home/user/miniconda3/envs/atec/bin/python` |
| Orin硬件环境 | 已有`/mnt/nvme/pi05/runtime/official-20260910/robot-venv/bin/python`，SDK 0.2.9.2 |
| 头部相机 | 既有`rtsp://192.168.168.168:8554/front`，25 FPS |
| 腕部相机 | 既有`realsense://260322275842`，30 FPS采图，策略数据约定仍为25 Hz |
| 连接方式 | SSH反向转发；两端策略端口只监听localhost:8026 |

客户端包带固定硬件辅助代码、配置和一帧离线样本；复用Orin现有SDK/相机依赖，不安装Torch或Depth Anything，也不替换既有机器人项目。`robot`未加`--execute`时只打印命令，不导入或连接硬件。

本版五个硬件辅助文件原样取自Orin已部署的`/mnt/nvme/pi05/runtime/official-20260910/openpi_v0.2.0/examples/airbot/`。项目快照在`vendor/airbot_orin_20260924/openpi/`，相邻`provenance.json`记录远端来源和逐文件SHA256；客户端导出清单继续记录所用快照文件及SHA256。它保留历史Orin实测的100 Hz频率传递和不追赶补发的控制循环。原有导出包保持原样；新客户端在预览中显示100 Hz，并在旧helper不支持该配置时拒绝连接机械臂。

## 1. 5090启动策略

在5090终端一运行；保持终端打开，用Ctrl+C停止服务：

```bash
cd /home/user/wang-sm/depth-mani
python3 scripts/deploy_airbot_paperbag.py serve
```

入口先核验推理包全部文件SHA，固定使用本次最佳权重；不自动选择其他实验或最终100k权重。

## 2. 将客户端复制到Orin并建立转发

首次安装，在5090终端二复制已经生成的客户端包。这里使用既有Orin地址；SSH认证由操作者完成。该目录与之前的纸袋/西瓜部署目录独立，已有同名副本时先保留并核查，不向其他部署目录覆盖文件。

```bash
cd /home/user/wang-sm/depth-mani
scp -r deploy/airbot-paperbag200-da2-100k-robot-budgetfix-20260924 \
  nvidia@100.68.24.27:/mnt/nvme/pi05/
python3 scripts/deploy_airbot_paperbag.py tunnel --robot-host nvidia@100.68.24.27
```

转发命令保持前台运行；它把Orin的127.0.0.1:8026转到5090的127.0.0.1:8026。若该端口已被其他程序占用，命令会报错，不替换已有进程。SSH使用已记录的主机身份，不关闭主机身份检查。

## 3. 在Orin检查通信和相机

在Orin本地终端或SSH终端运行：

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-100k-robot-budgetfix-20260924
sha256sum -c SHA256SUMS
python3 deploy.py check
python3 deploy.py probe
python3 deploy.py cameras
ip -brief link
```

`check`只检查依赖并列出CAN接口；`probe`用包内真实数据样本请求策略，不接触硬件；`cameras`只打开双相机，将PNG和检查记录保存到打印出的`runtime/airbot-deploy/cameras-*`目录。实际图像必须是正确的头部与腕部视角，不做224/256缩图、旋转或裁剪。摄像头地址/序列号来自先前采集设置，设备更换后需使用实际配置。

## 4. 人工执行机械臂策略

先确认物理从臂对应的CAN接口。下面的`can2`来自Orin历史SLCAN部署记录；USB重新枚举后接口名可能改变。`0 0 0 0 0 0 0.07`是既有启动器的七维reset目标，reset是运动目标，不是标零，需与当前机械臂零位和可达姿态一致。现场停止其他控制程序，并准备好急停。

先预览同一条命令：

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-100k-robot-budgetfix-20260924
python3 deploy.py robot --can-interface can2 \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25
```

确认参数后，显式执行：

```bash
python3 deploy.py robot --can-interface can2 \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25 --execute
```

客户端先核对模型SHA、prompt、25 Hz、horizon=8、相机映射和7维绝对动作，再连接硬件。连接时SDK会使能电机并保持测得的当前位置；在提示处按Enter后才reset并执行。运行25个策略步后等待下一轮；按`q`再Enter退出，不额外reset。运行中Ctrl+C停止新动作并进入SDK清理流程；程序会尝试断使能，现场需确认机械臂状态。

复位沿用SLOW速度设置：六关节π/3 rad/s、夹爪0.015 m/s，平滑复位轨迹至少2秒；策略执行沿用FAST设置：六关节π rad/s、夹爪0.03 m/s。100 Hz是底层发送频率，25 Hz仍是策略动作发布目标，每次8步预测最多执行前4个完整目标；这些频率不表示完整RGB推理能达到同样的闭环频率。

短程行为正常后，完整任务可将`--max-steps 25`改为`--max-steps 300`。这是策略步数，不是抓取次数，也不自动判定成功。运行记录保存到打印出的`runtime/airbot-deploy/robot-*`日志。

客户端会在执行一块动作前核验维度、有限值、关节限位及全部插值点；每个目标开始前检查“插值点数÷25 Hz”是否能放进剩余的2秒图像时效。若已完成部分目标而下一目标放不下，记录`chunk_replan`、丢弃余下动作并重新采图和推理，仅累计完成目标。若首目标也放不下，立即停止且本块不发送动作；相机停帧、通信超时、实际执行阻塞或非法动作仍中止。关节2仍保留原0.17 rad上限，训练数据中高于此值的目标不会被自动放行。G2沿用既有控制器的[0,0.072]m裁剪并明确记录警告；reset目标则要求全部处于限位内。

## 硬件直接接在5090时

无需Orin或SSH转发。5090终端一仍运行第1节服务；另一个5090终端使用相同客户端入口：

```bash
cd /home/user/wang-sm/depth-mani
python3 scripts/deploy_airbot_paperbag.py check
python3 scripts/deploy_airbot_paperbag.py cameras
python3 scripts/deploy_airbot_paperbag.py robot --can-interface can0 \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25 --execute
```

只有CAN和两路相机实际接在该主机且可访问时才使用此方式。可通过`--robot-python`、`--openpi-root`指定已经准备好的兼容环境；相机配置不同可复制`configs/airbot_paperbag_deploy.json`为新文件，用`--profile`显式选择，不改动带清单的原始文件。

## 验证范围

包完整性、固定权重及无硬件通信检查独立于抓取表现。当前最佳权重的首步动作误差仍高于保持当前状态的基线，不能据此宣称真机抓取已可靠。真实硬件相机、CAN、零位和抓取结果以现场运行记录为准。

本机已验证机器人Python环境经WebSocket调用真实76k权重：8×7动作与导出参考逐值一致，最大差异为0。首次冷请求约1.38秒，随后请求约100毫秒；这是同机通信实测，不包含Orin网络或真实相机采集。每次请求发送约7.44MB原始RGB，跨机器时使用稳定的高速连接，并以Orin上的`probe`实测为准。

2026-09-24取得本版辅助代码快照时，已通过SSH只读确认Orin现有SDK环境和D405 SDK序列号`260322275842`。当时未枚举到历史USB-SLCAN适配器，仅有DOWN状态的内置can0/can1，不能将它们直接当作从臂接口。本次快照与软件回归未连接、使能或操作机械臂；真实动作验收另行记录。
