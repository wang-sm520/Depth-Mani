# AIRBOT 易拉罐：RGB → Depth Anything → Orin

`pick can bd` 使用和红纸袋相同的 AIRBOT Play 原生接口：头部 RGB 为
`1920×1080`、腕部 RGB 为 `848×480`，状态和绝对动作均为六个关节弧度加夹爪米制开度。
`scripts/export_airbot_can_orin.py` 从新 checkpoint 读取 prompt、相机顺序、动作维度和 SHA256，
不会复用红纸袋的 checkpoint 身份。推理时 `airbot_depth.policy.AirbotDepthPolicy.infer_rgb`
按训练时固定的 Depth Anything V2 Small 配置处理两个 RGB 视角，得到每视角
`[relative_inverse_depth, validity_mask, 128, 128]`，再送入策略网络。

训练和审计完成后，在工作站生成模型包和部署 profile（命令中的路径按本次 run 修改）：

```bash
PY=/home/user/miniconda3/envs/atec/bin/python
$PY scripts/export_airbot_can_orin.py bundle \
  --checkpoint runs/airbot-can100-da2-50k-20260929/best.pt \
  --audit reports/airbot-can100-da2-50k-20260929-reference-audit/report.json \
  --output deploy/airbot-can100-da2-50k-20260929-model \
  --config configs/airbot_can_deploy.json \
  --profile-output deploy/airbot-can100-da2-50k-20260929.profile.json \
  --device cuda --threads 4
```

生成的 bundle 是可单独复制到 Orin 的 model-only 包，包含 `policy.pt`、冻结的
`airbot_depth/` 和 `depth_policy/` 源码、Depth Anything 的 `hf_hub/` 缓存、
`sample_observation.npz`、审计记录和 `serve.sh`。先在工作站用同一 bundle 做离线
RGB→深度→动作检查：

```bash
$PY scripts/export_airbot_can_orin.py validate \
  --checkpoint deploy/airbot-can100-da2-50k-20260929-model \
  --observation deploy/airbot-can100-da2-50k-20260929-model/sample_observation.npz \
  --output reports/airbot-can100-da2-50k-20260929-orin-offline.json \
  --device cuda --threads 4
```

报告中的 `depth_provenance`、`depth_shape` 和 `[8,7]` `actions` 是离线验收证据；
它不连接机械臂，也不代表真实抓取成功率。`inspect` 可在复制前核对 bundle 清单：

```bash
$PY scripts/export_airbot_can_orin.py inspect \
  --checkpoint deploy/airbot-can100-da2-50k-20260929-model
```

若需要和已有 Orin layered runtime 一样做目标平台数值校验，审计时需固定四个 episode
（每个取起始、中间、末尾，共 12 个样本；至少一个属于 validation），再导出 reference：

```bash
$PY -m airbot_depth.audit \
  --data data/airbot-can100-da2-20260929 \
  --run runs/airbot-can100-da2-50k-20260929 \
  --output reports/airbot-can100-da2-50k-20260929-reference-audit \
  --device cuda --roundtrip-episodes 0 3 50 99
```

```bash
$PY scripts/export_airbot_can_orin.py reference \
  --bundle deploy/airbot-can100-da2-50k-20260929-model \
  --audit reports/airbot-can100-da2-50k-20260929-reference-audit/report.json \
  --episode-indices 0 3 50 99 \
  --output deploy/airbot-can100-da2-50k-20260929-reference \
  --device cuda --threads 4
```

这里的四个索引必须和 audit 的 `roundtrip.episode_indices` 完全一致；审计没有保存这
四组时，先重新运行 audit 再导出。已有机器人客户端 exporter 也接受动态 profile：

```bash
$PY scripts/export_airbot_robot.py \
  --profile deploy/airbot-can100-da2-50k-20260929.profile.json \
  --openpi-root /home/user/wang-sm/pi0.5/openpi_v0.2.0 \
  --output deploy/airbot-can100-da2-50k-20260929-robot-orin
```

具备 model、robot 和 12 个 reference 后，可组装完整 Orin 包：

```bash
$PY scripts/export_airbot_can_orin.py orin \
  --model-bundle deploy/airbot-can100-da2-50k-20260929-model \
  --robot-bundle deploy/airbot-can100-da2-50k-20260929-robot-orin \
  --references deploy/airbot-can100-da2-50k-20260929-reference \
  --client-source /home/user/wang-sm/openpi/packages/openpi-client/src/openpi_client \
  --output deploy/airbot-can100-da2-50k-20260929-orin-can
```

把完整包复制到 Orin 后先核对 `SHA256SUMS`。使用与目标 JetPack/Torch 匹配的 Python，
先运行 `inspect`，再以固定 reference 做 `validate --no-activate` 或完整 validate；只有
报告 `status=passed` 且包、reference、Python 和 CUDA 运行时身份一致时才启动 `serve`：

```bash
cd /mnt/nvme/pi05/airbot-can100-da2-50k-20260929-orin-can
sha256sum -c SHA256SUMS
MODEL_PYTHON=/mnt/nvme/pi05/airbot-paperbag200-da2-orin-native-20260924/model-env/bin/python
ROBOT_PYTHON=/mnt/nvme/pi05/runtime/official-20260910/robot-venv/bin/python
NATIVE_LIB=/mnt/nvme/pi05/runtime/official-20260910/native/lib
python3 deploy.py inspect --python "$MODEL_PYTHON" --report runtime/inspect-can-20260929.json
python3 deploy.py validate --python "$MODEL_PYTHON" \
  --robot-python "$ROBOT_PYTHON" --native-library-dir "$NATIVE_LIB" \
  --report runtime/attestation-can-20260929.json
python3 deploy.py serve
```

`serve` 只监听本机 `127.0.0.1:8026`，不导入机械臂 SDK。相机和机器人客户端仍须在
另一个终端运行；`cameras` 只检查两路原生 RGB，`probe` 只发送记录观测，`robot` 默认
预览，确认 CAN 接口和 reset 动作后才显式追加 `--execute`。沿用已有客户端的 25 Hz
策略步、100 Hz 底层控制、每次最多执行 4 个目标和图像时效检查。

源数据中 joint2 有 807/19,701 帧高于现有客户端的 `+0.17 rad` 参考上限（最大约
`0.1764 rad`）。客户端会拒绝越界动作；部署前必须按现场零位和真实限位核对，不能
静默裁剪，也不能把离线动作误差当成抓取成功率。
