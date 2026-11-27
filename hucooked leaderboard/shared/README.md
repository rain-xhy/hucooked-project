# Shared 三人厨房榜单

五版使用同一环境修订和同一组 24 个评估种子，每局 300 步。点击方法名可查看模型参数、训练步数、训练局数、计时口径及文件入口。

| 方法 | 平均出餐数 | 平均团队得分 | 同步成功率 | 每局等待步数 | 入榜训练量 | Unity 回放 |
|---|---:|---:|---:|---:|---|---|
| [随机初始化 · 纯 PPO](multi-agent/runs/01-pure-ppo-380/README.md) | 19.00 | 380.00 | 21.74% | 45.00 | 144 轮 PPO · 221,184 环境步 | [打开文件](multi-agent/runs/01-pure-ppo-380/replay/shared.ocrep.json) |
| [420 规则策略 · BC](multi-agent/runs/02-rule-bc-420/README.md) | 21.00 | 420.00 | 18.03% | 50.00 | 60 局示范 · 25 个 BC epoch/网络 | [打开文件](multi-agent/runs/02-rule-bc-420/replay/shared.ocrep.json) |
| [440 周期规则策略 · BC](multi-agent/runs/03-rule-bc-440/README.md) | 22.00 | 440.00 | 100.00% | 11.00 | 1 局示范 · 291/293 次 BC 更新 | [打开文件](multi-agent/runs/03-rule-bc-440/replay/shared.ocrep.json) |
| [420 BC 起点 · 规划器引导的离线 Q](multi-agent/runs/04-planner-q-440/README.md) | 22.00 | 440.00 | 100.00% | 11.00 | 4 段短轨迹及局部分支 · 1,200 次 Q 更新 | [打开文件](multi-agent/runs/04-planner-q-440/replay/shared.ocrep.json) |
| [420 BC 技能分段 · 高层 PPO](multi-agent/runs/05-segment-ppo-440/README.md) | 22.00 | 440.00 | 50.00% | 12.00 | 23 轮高层 PPO · 753,664 局（全体） | [打开文件](multi-agent/runs/05-segment-ppo-440/replay/shared.ocrep.json) |

同步成功率按有效尝试计算。训练量依方法分别记录示范环境步、在线环境步及优化器更新数。第四、第五版使用的上游 420 权重统一保存在第二版。每份 Unity 文件包含种子 84031 的 300 个动作步与 301 帧。

## 订单指标的计算口径

本榜单按下列公式计算主仓库列出的两项订单指标。每版合并全部 24 局的逐单记录后计算。点击方法名可查看分子、分母和累计交付步数。[主仓库指标表](https://github.com/rain-xhy/hucooked-project/blob/fc486ec4d01388c46356ed17419249fc6816b22b/README.md#实验成绩)

| 方法 | 已交付 / 已发布（24 局合计） | 订单完成率 ↑ | 平均交付步数 ↓ |
|---|---:|---:|---:|
| [随机初始化 · 纯 PPO](multi-agent/runs/01-pure-ppo-380/scores.json) | 456 / 504 | 90.48% | 29.47 |
| [420 规则策略 · BC](multi-agent/runs/02-rule-bc-420/scores.json) | 504 / 552 | 91.30% | 27.71 |
| [440 周期规则策略 · BC](multi-agent/runs/03-rule-bc-440/scores.json) | 528 / 576 | 91.67% | 26.59 |
| [420 BC 起点 · 规划器引导的离线 Q](multi-agent/runs/04-planner-q-440/scores.json) | 528 / 552 | 95.65% | 26.64 |
| [420 BC 技能分段 · 高层 PPO](multi-agent/runs/05-segment-ppo-440/scores.json) | 528 / 576 | 91.67% | 25.73 |

- **订单完成率** = 已交付订单总数 ÷ 已发布订单总数。以终局订单表为准，`released_at >= 0` 表示已发布，`status == "served"` 表示已交付。开局发布的订单和截止时仍在制作的订单均计入分母。
- **平均交付步数** = 所有已交付订单的 `completed_at - released_at` 之和 ÷ 已交付订单总数，单位为步/单。普通单与协作单一起统计。例如，开局第 0 步发布、第 17 步送达的一单计 17 步。

三版 440 每局都交付了 22 单。连续订单模式最多同时开放两单，第 300 步结束时停止补单。离线 Q 每局最后一单恰在第 300 步送达，本局累计发布 23 单。周期 BC 和分段 PPO 分别在第 299、289 步完成最后一单，环境当步补单，本局累计发布 24 单。因此，离线 Q 的完成率受终局补单规则影响。三版中，分段 PPO 的订单从发布到交付用时最短，平均为 25.73 步。[环境补单逻辑](https://github.com/rain-xhy/hucooked/blob/de37e0818c2e76311e105ab8689e5162bd84133f/src/overcooked_ai_py/cooperative/shared/kitchen.py#L115-L120)
