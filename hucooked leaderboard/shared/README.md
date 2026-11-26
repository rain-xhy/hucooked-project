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
