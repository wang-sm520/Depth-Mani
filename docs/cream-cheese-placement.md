# 奶酪盒中心位置范围推导

官方任务为 `libero_goal/put_the_cream_cheese_in_the_bowl`，语言为
`Put the cream cheese on the bowl`，成功判据为
`On(cream_cheese_1, akita_black_bowl_1)`。原范围配置为
`configs/cream_cheese_on.json`；仅供三倍范围评估的配置与场景分别为
`configs/cream_cheese_3x_on_eval.json`、`configs/evaluation/cream_cheese_3x.bddl`。
这些文件不修改上游 LIBERO，也不改变采集或官方原范围评估。
早期不带 `_on` 的配置属于隔离的错误指令试跑，不用于本实验。

## 原范围

官方 BDDL 的 `cream_cheese_region` 矩形为
`(x_min,y_min,x_max,y_max)=(-0.06,0.12,-0.04,0.14)` m。
奶酪盒 XML 的 `horizontal_radius_site` 位于 `(0.03,0.03,0)` m，
`MujocoXMLObject.horizontal_radius` **只取第一个坐标**，因此水平半径
`r=0.03` m，不是 `sqrt(0.03^2+0.03^2)`。
该任务选择 `TableRegionSampler`，默认启用
`ensure_object_boundary_in_range=True`、`ensure_valid_placement=True`。
采样器分别调用旧版 `np.random.uniform(low=x_min+r, high=x_max-r)`
和相应的 y 范围。原范围的参数实际上是反向区间：

| 轴 | 原始 BDDL | `uniform(low, high)` 实参 | 候选中心支持集 |
| --- | --- | --- | --- |
| x | [-0.06, -0.04] | (-0.03, -0.07) | [-0.07, -0.03] |
| y | [0.12, 0.14] | (0.15, 0.11) | [0.11, 0.15] |

现用 NumPy 1.26.4 的旧版 `np.random.uniform` 接受上述反向实参，
候选中心宽度仍为每轴 **0.04 m**，中心为 `(-0.05,0.13)` m。
`np.random.default_rng().uniform` 对反向实参会抛错，不能用它
模拟上游的实际采样行为。世界 x/y 平移为零，因为 tabletop
`workspace_offset=(0,0,0.90)` m。

## 三倍评估范围

目标中心保持 `(-0.05,0.13)` m，每轴候选总宽变成 `0.12` m：

| 轴 | 原候选中心 | 三倍候选中心 | 新 BDDL 边界 |
| --- | --- | --- | --- |
| x | [-0.07, -0.03] | [-0.11, 0.01] | [-0.14, 0.04] |
| y | [0.11, 0.15] | [0.07, 0.19] | [0.04, 0.22] |

新 BDDL 的唯一变更是奶酪盒矩形 `(-0.14,0.04,0.04,0.22)` m。
经同一个采样器各边扣除 `r=0.03` m 后，`uniform` 得到有序
`low/high`，分别是 `(-0.11,0.01)` 与 `(0.07,0.19)` m。
这是每轴宽度三倍、候选矩形面积九倍，不是实际成功概率九倍。
本地评估配置对 BDDL 做 SHA256 固定，对奶酪盒采样中心做范围校验。

## 分布与合法性边界

两个场景中的桌面、全部其他物体、朝向、资产、相机和目标谓词相同。
奶酪盒的 x 轴旋转固定为 `0`；目标碗继续按其官方区域随机化。
放置采样还会对先前放置的物体做水平半径碰撞拒绝，最多尝试 5000 次，
不合法时直接报错。因此上述界限只是**碰撞拒绝前的候选中心支持集**，
不能声称最终初态在矩形内严格均匀或完整覆盖边缘。
正式 bank 应检查原始中心坐标、稳定后坐标、与碗的相对位置、
所有物体的姿态和生成失败记录。三倍范围配置仅在
`evaluation_only=true` 时加载，不允许用于训练示范。
