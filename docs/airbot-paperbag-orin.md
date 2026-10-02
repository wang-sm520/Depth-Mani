# AIRBOT 红纸袋：Orin AGX 本机部署

将Depth Anything V2 Small、76k最佳策略和机械臂客户端放在Orin AGX的NVMe上运行。模型来自200条、25 Hz双相机示范的100k训练；推理与控制分别使用两个Python环境，通过Orin本机127.0.0.1:8026通信。

本版修复动作插值耗尽图像有效期后退出的问题：每次最多执行4个完整目标；若下一目标无法在图像的2秒有效期内完成，丢弃本块余下动作，重新采图和推理。25 Hz插值节拍、100 Hz底层控制、插值步长、速度和限位保持原值。

目标目录：`/mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924`。包内含模型权重、固定模型缓存、现场硬件辅助代码快照、12组校验观测和命令入口。复用旧版已安装的Orin模型环境；旧包及服务独立保留，切换前在旧服务终端按Ctrl+C，再从第4节启动本版。

旧版2026-09-24的12/12固定观测校验、本机通信和双相机连续读帧均已通过；当时双视角深度加策略计算均值约0.58秒，本机请求约0.86秒，尚未实现25 Hz完整闭环。修复版的独立验收以本目录`runtime/active.json`、`runtime/attestation-budgetfix-20260924.json`及`runtime/installation-budgetfix-20260924.json`为准。已完成本版验收时，日常使用直接从第4节开始，不必重装环境。

当前包的生成状态和Orin安装状态分开记录：顶层`manifest.json`验证文件完整性；只有在目标机生成通过的`runtime/attestation-*.json`和`runtime/active.json`后，才能启动本地服务。组装完成不等于已经在Orin完成验证。

## 1. 安装前检查

进入已经复制到NVMe的新目录：

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924
sha256sum -c SHA256SUMS
python3 deploy.py inspect
```

`inspect`记录板卡型号、L4T、NVMe真实挂载、可用Python和实际Torch CUDA小张量运算结果，不连接机械臂。保存路径会打印在终端。新旧Orin环境路径都会检查，历史JAX成功记录不能替代Torch CUDA检查。

## 2. 复用本机模型环境

本次直接使用旧版已验证的模型Python，无需运行setup。它通过独立的`cuda-base`层复用NVMe上`/mnt/nvme/home/miniconda3/envs/anygrasp`的NVIDIA Torch 2.4.0a0 nv24.05、系统CUDA 12.2及cuDNN 8.9.4，并使用为其构建的官方torchvision 0.18.1。已有OpenPI环境实测是CPU版Torch，不能用作CUDA基础环境。

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924
AIRBOT_MODEL_PYTHON=/mnt/nvme/pi05/airbot-paperbag200-da2-orin-native-20260924/model-env/bin/python
python3 deploy.py inspect --python "$AIRBOT_MODEL_PYTHON"
```

需保留旧版`model-env`、`/mnt/nvme/pi05/airbot-paperbag200-da2-orin-20260924/cuda-base`和`anygrasp`三个运行依赖。原环境复核记录在旧包`runtime/env-recovery-20260924T150040Z-9ca6c6f1.json`；构建来源、固定commit、wheel哈希和CUDA记录位于`/mnt/nvme/pi05/airbot-paperbag200-da2-orin-20260924/runtime/vision-bootstrap-v0181/`。Torch alpha版没有`register_fake` API，因此使用支持其`impl_abstract`接口的torchvision源码；未伪造Torch版本。

如果没有候选通过CUDA运算检查，应先根据实际JetPack/L4T准备匹配的Jetson Torch；该入口不会从PyPI自动换装一个未经验证的Torch，也不会静默退回CPU。

## 3. 验证Orin上的深度与动作

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924
python3 deploy.py validate \
  --python /mnt/nvme/pi05/airbot-paperbag200-da2-orin-native-20260924/model-env/bin/python \
  --robot-python /mnt/nvme/pi05/runtime/official-20260910/robot-venv/bin/python \
  --native-library-dir /mnt/nvme/pi05/runtime/official-20260910/native/lib \
  --report runtime/attestation-budgetfix-20260924.json
```

验证覆盖12个固定原生RGB/状态观测，包含独立验证轨迹。它分别比较实际送入深度模型的输入张量、深度与mask、相同深度输入的策略动作，以及完整RGB→深度→动作。原权重和训练来源信息保持原样；目标库版本和硬件信息另存入校验记录。

参考生成与本机服务均固定cuDNN和矩阵乘的TF32为关闭，与训练审计一致。动作比较使用`references/`内的对应观测基准；原模型包的`bundle-validation.json`保留历史导出设置下的结果。

固定门槛为：输入张量最大绝对差`1e-6`；mask完全一致；归一化深度有效像素最大差0.005、平均差0.0005；相同深度下六关节/夹爪动作最大差分别为0.0001 rad/0.00001 m；完整RGB路径分别为0.002 rad/0.0002 m。所有相对容差为0，不在失败后自动放宽。通过这些数值检查不代表实机抓取成功率。

若现场SDK路径不同，修改上面的`--robot-python`和`--native-library-dir`。新报告不覆盖旧报告；已有同名报告时改用新的报告文件名。环境、模型或代码变化后，旧校验会失效，需要重新验证。验证通过后，`active.json`记录复用的模型Python，后续serve无需再指定它。

## 4. 启动策略服务

在Orin终端一运行，等终端显示监听127.0.0.1:8026：

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924
python3 deploy.py serve
```

入口重新核对校验记录与当前环境，并在开始监听前重验一组固定观测、预热GPU。模型及深度缓存均从本机读取。此部署模式运行期间不依赖5090或SSH转发。

## 5. 检查相机并人工执行

在Orin终端二运行：

```bash
cd /mnt/nvme/pi05/airbot-paperbag200-da2-orin-budgetfix-20260924
python3 deploy.py check
python3 deploy.py probe
python3 deploy.py cameras
ip -brief link
```

`probe`使用包内观测验证本机通信；`cameras`只采集相机，不使能电机。沿用头部`rtsp://192.168.168.168:8554/front`和D405序列号`260322275842`，输出原尺寸RGB：1920×1080与848×480。

2026-09-24本次排查已确认从臂USB CAN枚举为`can2`且UP，`slcan@0.service`运行。重插后用`ip -brief link`确认实际接口名；内置`can0/can1`不能直接当作从臂接口。原SLCAN配置是经典CAN的1 Mbit/s，串口速率3 Mbaud；不要对它套用内置mttcan初始化命令。

确认从臂接口及reset路径后，先预览，再显式执行。以下`can2`必须替换成现场确认的接口；七维reset沿用已记录的零关节角与0.07 m夹爪开度：

```bash
AIRBOT_CAN=can2
python3 deploy.py robot --can-interface "$AIRBOT_CAN" \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25

python3 deploy.py robot --can-interface "$AIRBOT_CAN" \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25 --execute
```

未加`--execute`只预览。加上后，程序先核对模型及本机校验身份，再连接并使能电机保持当前位置；按Enter后才reset并运行。到达步数后等待下一轮，`q`再Enter退出，不追加reset；Ctrl+C停止新动作并尝试SDK清理，需现场确认电机状态。

策略固定输出8步绝对位置，每次最多执行4个完整目标，再重新观测；六关节单位rad，夹爪单位m。每个目标开始前按“插值点数÷25 Hz”检查剩余图像时效；放不下下一目标时，日志出现`chunk_replan`，丢弃本块余下目标后重新采图和推理，仅已完成目标计入步数。若首目标就无法在有效期内完成，则在该块发出任何动作前停止；相机停帧、通信超时或执行意外阻塞仍停止。

25 Hz是插值点发布节拍，推理及插值会增加实际时间。底层电机命令采用现场记录已验证的100 Hz，并沿用延迟后不突发补发的实现；这两个频率含义不同。保留原关节限位和G2有日志的裁剪行为，越界停止。短程行为正常后，可将`--max-steps 25`改为`--max-steps 2500`，它表示实际完成的策略目标数，不是抓取次数。

本次用户日志中的G2裁剪提示不是退出原因；SDK也明确报告电机1至3不支持显式disable，进程退出不等于这些电机已经断使能，停止后需现场确认。

固定prompt为`pick up the red paper bag and hold in the reset position`，由策略提供；该单任务模型不支持任意自然语言任务。

## 本次交付边界

此包保留原模型审计与参考生成记录，旧导出包也继续保留。真实Orin安装及CUDA数值校验以目标机器的`runtime/active.json`、校验报告和安装记录为准；打包时manifest中的准备状态不会因此被改写。人工执行前需恢复采集时的相机安装和任务摆放。此次修复部署和软件验证不执行机械臂动作；用户之前的实际运行已完成目标0至26后报超时，不能将那次运行记作“未执行”，也不能据此认定抓取成功。
