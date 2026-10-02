# 会话交接：Depth/RGB 机器人操作实验

## 新窗口先做什么

1. 本文件是历史事实与工程约束的摘要，不是完整聊天记录，也不自动授权启动实验。
2. 若用户要求执行奶酪盒任务，再读取同目录的 `next-task-cream-cheese.md`；该文件是下一任务协议。
3. 先检查真实文件、后台进程、GPU和磁盘，再制定计划。不要依据历史快照重复启动任务。
4. `reports/progress.md` 有多段历史“最新/运行中”标题；以最新完成记录、真实状态文件和结果审计为准。

## 研究目标和用户偏好

- 工作目录：`/home/user/wang-sm/depth-mani`；与用户中文沟通。
- 验证小规模、随机初始化的 Depth + 机器人状态 + 语言策略能否完成操作任务，RGB作为独立对照。
- 学生不用视觉/语言预训练、不融合RGB-D、不接收物体真实位置等特权状态。
- 官方 `pi05_libero` 作教师；不是 pi05 base。教师使用双视角RGB、机器人状态和指令。
- 用户关心真实闭环成功率、录像、训练阶段对比、初始位置分布和可复现性；不要只报告动作loss。
- “1w/5w轮”在本项目指10000/50000 optimizer step，不是epoch。
- 必须明确区分任务timeout、程序异常、基础设施中断；保留失败和异常，不挑成功样本汇报。
- 不擅自恢复已暂停的酒瓶教师采集；新任务只有收到用户执行授权后再启动。

## 酒瓶实验已完成，不要重复执行

任务：`libero_goal/put_the_wine_bottle_on_top_of_the_cabinet`。

最新一轮用同一批300条成功训练轨迹（35个独立训练初态、27279个正式动作帧）和7条验证示范，
两模型seed=0从零训练到50k，保留10k/20k/30k/40k/50k权重。
每种分布200个冻结新随机初态、两模型五阶段共4000次正式闭环。
北京时间2026-09-24 01:18完成审计和绘图，服务正常退出。

| step | 原范围Depth/200 | 原范围RGB/200 | 10cm Depth/200 | 10cm RGB/200 |
| ---: | ---: | ---: | ---: | ---: |
| 10000 | 197 | 196 | 132 | 116 |
| 20000 | 198 | 198 | 125 | 108 |
| 30000 | 198 | 197 | 125 | 113 |
| 40000 | 197 | 197 | 118 | 112 |
| 50000 | 197 | 197 | 119 | 109 |

- 模型及正式轨迹：`runs/wine300-50k-20260923/`。
- 结果/CSV/JSON/逐组审计：`reports/wine300-50k-20260923/`。
- 图片：`reports/figures/wine300-50k-20260923/`。
- 协议：`configs/wine50k_experiment.json`、`reports/wine50k-protocol.md`。
- 后台状态：`runtime/wine300-50k-20260923-state.json`。
- 服务：`depth-mani-wine50k-20260923.service`，最后核查inactive、Result=success。
- 另有5条并发启动故障引起的中断轨迹，已原样归档，不计作策略timeout；没有丢弃成功/超时结果。
  证据：`reports/wine50k-startup-recovery.json`。

解释边界：后期训练loss明显降低，验证动作loss基本停滞或略降；原范围成功率稳定，扩展范围没有随训练持续提升。
可能涉及窄分布适配、动作loss与闭环指标不一致、缺少纠错数据；尚未证明具体失败机制或“训练越久必然越差”。
只有一个训练seed、一个任务，不推广成一般模态优劣结论。

### 较早结果不要与最新一轮混淆

- 旧300条训练的Depth `best.pt` 是5500 step、RGB `best.pt` 是6000 step；旧 `latest.pt` 是10000 step。
- 旧Depth best：官方固定50初态49/50，原范围新随机100初态100/100，10cm新随机100初态69/100。
- 旧10k latest的同组10cm 100初态：Depth66/100、RGB52/100。
- 最新50k实验是重新训练和另一组200初态，不能将这些数拼接成同一条曲线。
- 旧500条酒瓶扩充由用户暂停，磁盘上多余示范不能偷偷加入固定300条训练子集。

## 工程环境与可复用代码

- 项目仿真：`bash scripts/sim_cpu.sh ...`，使用项目`.venv`、本地OSMesa和CPU；原图256、模型输入128。
- 项目`.venv`曾确认torch 2.7.1+cu126不能在RTX5090执行CUDA内核；不要直接用它训练GPU。
- 已验证GPU训练环境：`/home/user/miniconda3/envs/atec/bin/python`，torch 2.7.0+cu128。使用前重新检查。
- 上游：`/home/user/wang-sm/openpi`，LIBERO在其`third_party/libero`；不修改上游或升级共享环境。
- 教师权重：`/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_libero`。
- 教师入口：`scripts/serve_teacher.py`、`scripts/run_teacher_cpu_batch.sh`；教师环境为OpenPI自己的`.venv`。
  原批处理脚本启动的是CPU教师。先检查端口和进程归属，不接管无关服务。
- 模型是CNN图像编码器、状态MLP、Embedding/GRU语言分支和融合MLP，约6.67M参数。
- AdamW、lr=3e-4、weight_decay=1e-4、batch=64、horizon=8、seed=0。
- 控制OSC_POSE、20Hz、稳定10步、任务最多300步；学生预测8步执行前4步重规划，原教师执行前5步。
- `depth_policy/train.py`支持`--checkpoint-every 10000`；保留`step_010000.pt`等，另外存latest/best。
- `depth_policy/evaluate_suite.py`支持冻结随机bank、配对评估、`--compact`、磁盘下限和输出身份检查。
- `depth_policy/episodes.py`：schema1全量训练数据；schema2仅evaluation-only紧凑记录。
  后者保留全部动作/状态/时间、图像hash、首个策略观测、初末状态及XML；不等于逐帧保存全部图像。
- `depth_policy/replay.py`支持两类记录的精确回放和视频导出；紧凑模式会核对RGB-D hash。
- `depth_policy/simulation.py::configure_libero`已修复共享配置截断竞争，采用原子替换；必须保留。
- `scripts/wine50k_experiment.py`及`wine50k_report.py`可参考，但硬编码酒瓶路径、旧数据审计和测试数量，
  不要不加修改就用于奶酪盒，也不要覆盖已完成实验。
- 原`collect`/manifest流程以固定官方初态为基础。奶酪盒要求300个不同的随机训练初态，需检查并实现相应入口，
  不能简单循环旧固定初态，也不能把`evaluate_suite --teacher`产生的random-test/evaluation-only轨迹直接当训练集。
- 测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 bash scripts/sim_cpu.sh -m pytest -q`。
  最近全套30 passed、1 skipped（CUDA测试）；任何修改后按实际结果重新验证。
- 不进行git提交/分支，不删除旧实验，不停止无关服务，不以扩大数据量为由默认恢复旧采集。
- 本地图片/视频回复使用绝对路径Markdown嵌入；代码文件引用用绝对路径。

## 已核对的奶酪盒任务与重要差异

下一任务规格在`docs/next-task-cream-cheese.md`，本交接阶段没有启动该实验。

官方文件：
`/home/user/wang-sm/openpi/third_party/libero/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl`。

- 任务名虽含`in`，当前语言为`Put the cream cheese on the bowl`。
- 当前成功目标为`On cream_cheese_1 akita_black_bowl_1`，不要擅自改为`In`。
- 原奶酪盒BDDL名义x范围[-0.06,-0.04]m，y范围[0.12,0.14]m。
  这不是已验证的有效中心采样范围！必须检查奶酪盒自己的半径/边界/旋转/拒绝采样规则。
- 酒瓶原任务曾存在小区域扣半径后反向区间，有效中心总宽约3cm；这是酒瓶特例，不可直接套到奶酪盒。
- 下一实验的扩大方式为仅奶酪盒中心x/y有效总宽各乘3；碗和其他物体仍按官方原范围随机化。
- 奶酪盒拟用300个不同初态的成功示范，而酒瓶旧300条只有35个初态。
  因此未来跨任务比较还同时改变了训练初态多样性，不能归因于任务或模态单一因素。

## 下一窗口启动提示

在同一远程工作区打开新会话后，要求先读取本文件和`docs/next-task-cream-cheese.md`。
确认用户授权执行后，复述关键设置、检查真实环境，再建立新任务计划。
模型选择由客户端控制；不要声称仅靠prompt就切换了当前会话模型。
