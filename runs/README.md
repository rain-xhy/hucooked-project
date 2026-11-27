# 模型权重

已发布的 checkpoint 按方法存放在此目录。文件来源、大小和 SHA-256 见 [checkpoints.json](checkpoints.json)。评测时使用这里的权重路径，源码仍保留在 `src/`。

| 方法 | 发布权重 | 用途 |
| --- | --- | --- |
| NestRL L1 | `nest_rl/l1/policy_000025.pt`、`policy_000030.pt`、`policy_000040.pt`、`policy_000200.pt` | L2 的四成员队友池；U40 为 L2 初始化权重，U200 为 L1 末期权重 |
| NestRL L2 | `nest_rl/l2/policy_000200.pt` | L2 末期权重，也是 L3 初始化权重 |
| NestRL L3 | `nest_rl/l3/policy_000025.pt` | L3 入选权重，也是 L4 初始化权重 |
| NestRL L4 | `nest_rl/l4/best.pt` | L4 U165 入选权重 |
| MAT（tzw） | `mat/fixed-8.pt`、`mat/shared-imitation.pt`、`mat/shared-ppo-best.pt` | 固定订单旧权重、连续订单示范微调、PPO U12 入选权重 |
| HAHA SharedKitchen | `haha/shared/multi-agent/runs/<实验名>/train/best.pt` | 五个 SharedKitchen 榜单版本 |
| HAHA 历史版本 | `haha/{original,v1,v2,v3}/{multi-agent,single-agent}/runs/20261124-fresh/` | 保留原有 `train/` 和 `worker-pretrain/` 权重；需使用各自匹配的环境版本 |

HAHA 的五个 SharedKitchen 实验名为 `01-pure-ppo-380`、`02-rule-bc-420`、`03-rule-bc-440`、`04-planner-q-440`、`05-segment-ppo-440`。训练证据、评估数据和回放继续位于 `src/haha/leaderboard/`，各实验 README 的权重链接已指向这里。

L1–L4 的其他快照及 `latest.pt` 完整续训状态只在本地保存，由 `.gitignore` 忽略。D 盘原训练目录保留原文件；本目录中的副本已逐文件校验 SHA-256。`checkpoint_inventory.local.json` 是完整本地清单，也不加入 Git。发布的推理 checkpoint 不等同于含优化器及采样状态的续训快照。

STQ 的 26 组权重仍通过 [GitHub Release](https://github.com/rain-xhy/hucooked-project/releases/tag/stq-shared-final-20261127) 分发，不迁入此目录。ACAC 和 E3T 尚未收齐本地模型文件。
