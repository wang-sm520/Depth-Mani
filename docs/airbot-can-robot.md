# AIRBOT 易拉罐客户端

此客户端属于易拉罐 100 条示范、50,000 步训练的深度策略包，固定使用验证集选择的
34,000 步模型，prompt 为 `pick can bd`。头部与腕部 RGB 在 Orin 本机经过
Depth Anything V2 Small，再进入策略网络；模型输出 8×7 绝对位置，每次最多执行 4 个目标。

从完整包顶层启动，不要在本子目录运行纸袋的模型服务：

```bash
cd /mnt/nvme/pi05/airbot-can100-da2-50k-20260929-orin-can
systemctl --user start airbot-can-depth.service
python3 deploy.py probe
python3 deploy.py cameras
python3 deploy.py robot --can-interface can2 \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 25 --execute
```

`can2` 与 reset 参数沿用纸袋部署；连接时使能并保持当前位置，按 Enter 后复位和运行。
六关节单位 rad、夹爪单位 m。底层控制 100 Hz，插值节拍 25 Hz，推理耗时另计。
硬件辅助代码与纸袋 Orin budgetfix 包逐文件一致，保留图像过期后的重新观测行为和原有限位。
去掉 `--execute` 仅预览。源示范 joint2 最大为 0.1764 rad，部分值高于当前 0.17 rad 上限；
运行仍保留越界拒绝，需现场核验零位与限位。
