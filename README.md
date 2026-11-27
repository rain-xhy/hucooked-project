# 三人共享厨房 ACAC 协作方法

三名厨师在同一厨房完成连续混合订单。普通菜出锅由一名厨师操作，协作菜要求两名厨师在锅的左右操作位同一步交互。第三名厨师可并行切配菜、准备餐盘或送餐。三个角色使用相同的动作接口。

## 关键代码

| 目录 | 内容 |
| --- | --- |
| [policy](reproduction/acac_ours/policy/) | Actor 用公开观测和历史特征预测队友到位情况，将预测接入宏动作评分。开局分工模块给可行的角色排列打分。 |
| [macro_actions](reproduction/acac_ours/macro_actions/) | 6 个基础动作、11 个工位宏动作和连续切菜。每位厨师在自己的宏动作结束后重新选择。 |
| [training](reproduction/acac_ours/training/) | 团队训练奖励的组成及工序进展的计算。 |
| [evaluation](reproduction/acac_ours/evaluation/) | 从每局最后一步的 `info["metrics"]` 计算四项汇总指标。 |

环境通过仓库的 [hucooked 子模块](https://github.com/rain-xhy/hucooked/tree/508bcbf066216acb7bb66bdd2ecf9148bf73c7c1) 提供。任务规则、初始状态和渲染使用三人共享厨房环境。

## 独立评估结果

| 模型和场景 | 局数 | 平均出餐数 | 平均团队得分 | 同步成功率 | 平均同步等待步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours ＋学习开局分工，原图连续混合订单 | 96 | 21.00 | 420.00 | 30.05% | 25.51 |
| Ours ＋学习开局分工，同地图开局站位扰动 | 96 | 20.49 | 409.79 | 28.24% | 25.16 |
| ACAC 基线，原图连续混合订单参考 | 128 | 20.68 | 413.59 | 25.90% | 29.08 |

Ours 检查点为 13,162,308 步，ACAC 基线检查点为 12,750,000 步。基线参考使用另一批评估种子，表中两种方法的数据用于描述各自表现。两种 Ours 评估场景共用一张地图，区别在初始站位设置。每份送达餐计 20 分。同步成功率按整组的双人出锅成功次数除以有合格同步尝试的物理步数。等待按每局联合菜已可出锅但该步未成功的步数计。

[下载汇总 CSV](reproduction/acac_ours/results/mixed_orders_summary.csv)

## 运行演示

[三人共享厨房回放 MP4](reproduction/acac_ours/demo/shared_kitchen_replay.mp4)使用环境内置规则队伍。视频显示一局固定混合订单的运行过程，普通菜和协作菜各完成 4 份，得到 160 分。上表来自单独的学习模型评估。

| 输入 | 中间处理 | 输出 |
| --- | --- | --- |
| [初始厨房和订单](reproduction/acac_ours/demo/input.png) | [双人协作工序](reproduction/acac_ours/demo/processing.png) | [送餐和最终得分](reproduction/acac_ours/demo/output.png) |

小组的 [Unity 3D 回放器](https://github.com/LynxPeng/Better-replay-frontend-of-overcook-ai/tree/dc3a7f41c64a22e893d18504c7064d569337bbf4) 可从对应子模块查看。
