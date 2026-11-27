# V3 训练交接：BC 初始化、固定四阶段 PPO

更新：2026-11-24 20:49（北京时间）。以下训练进度是此时的快照，实时状态以运行目录中的日志、检查点和进程为准。

## 接手先看

当前正在运行的是 **从已有 BC 权重开始的 PPO**，不是随机初始化。运行名为 `counter-fixed-504x800-mb1600-bc-seed1-ppo-seed1`，PID 文件记录为 45536；接手时应重新检查进程命令行，不能只凭 PID 判断。

- 504 个 CUDA 厨房，每局 400 步，每厨房每轮采 800 步。
- mini-batch 1600，4 个 PPO epoch；每阶段固定 10 轮，四阶段共 40 轮。
- 前三阶段目标学习率 0.0003，联合微调 0.0001。
- 从 `runs/counter-bc-shaped-seed1/bc/bc_best.pt` 加载模型，Adam 和课程进度重新开始。
- PPO 期间继续使用示范 BC 损失，系数 0.1；过程奖励启用。
- 不按成绩提前晋级，不因旧岗位退步而终止阶段。KL 限制仍可提前结束一轮内剩余的梯度更新，不会减少计划中的采样轮数。

最后核对时处于左侧阶段，第 9/10 轮，全局第 9/40 轮，累计 3,628,800 环境步。训练进程存活，stderr 无报错。左侧验证表现有波动，尚不能认定训练稳定成功。

项目根目录：`D:\Projects\DRLExp\hucooked-project`。
当前运行目录：[counter-fixed-504x800-mb1600-bc-seed1](runs/counter-fixed-504x800-mb1600-bc-seed1/)。
面板：[Trackio，端口 7864](http://127.0.0.1:7864/?project=hucooked-v3-cooperation)。

## 当前问题与已核对的结果

用户关心左侧学不会或退步，以及右侧得分是否主要来自规则队友。判断时必须按学习岗位与等待对照比较，不能只看团队总均分。

当前运行的 45 局验证，每个岗位 15 局，使用同一套验证种子。下表均为原生交付奖励的团队均分，不包含过程奖励。

| 检查点/对照 | 总均分 | 左侧岗位 | 右侧主厨岗位 | 右侧传菜岗位 |
|---|---:|---:|---:|---:|
| BC 初始模型 | 160.44 | 80.67 | 196.67 | 204.00 |
| 学习者始终等待 | 158.89 | 0.00 | 181.33 | 295.33 |
| 全脚本 | 320.00 | 320.00 | 320.00 | 320.00 |
| PPO 第 2 轮 | 182.89 | 81.33 | 260.00 | 207.33 |
| PPO 第 4 轮 | 204.22 | 102.00 | 282.67 | 228.00 |
| PPO 第 6 轮 | 198.00 | 124.67 | 230.00 | 239.33 |
| PPO 第 8 轮 | 184.89 | 52.00 | 269.33 | 233.33 |

来源：当前运行的 `validation-bc.json`、`validation-wait.json`、`validation-script.json` 和 `validation/002.json` 至 `008.json`。本次复用了同配置的已有 BC/等待/全脚本基线缓存；BC 文件 SHA256 与原记录一致，未重新跑一遍基线。上述 45 局验证不是 450 局独立测试，不能与旧报告的 450 局均分直接混用。

第 6 到第 8 轮，左侧从 124.67 降到 52，说明加载 BC 并不能保证 PPO 保留原有技能；回退原因尚未定位。下一步应比较这两个检查点的同种子左侧轨迹，逐项看取料、放板、连续切菜、取走、交接、投锅和交付。当前没有证据足以把原因归为学习率、采样长度或某个单独损失。

右侧等待对照本身能得到较高分数。模型在右侧时，左侧规则厨师仍供给蔬菜，另一名右侧规则厨师也能加工。模型在左侧时，唯一左厨控制蔬菜来源，失效会阻断整个厨房。因此右侧团队分数较高不等于右侧模型贡献高。

## 不要混淆这几次训练

| 运行目录（均位于 runs 下） | 初始权重 | BC 损失系数 | 状态/用途 |
|---|---|---:|---|
| counter-fixed-504x800-lr0003-seed1 | BC | 0.1 | 历史固定课程，mini-batch 900，已停止 |
| counter-fixed-504x2000-scratch-seed1 | 随机 | 0 | 已停止；当时明确选择跳过 BC |
| counter-fixed-504x800-mb1600-scratch-seed1 | 随机 | 0 | 已停止；曾沿用跳过 BC 的设置 |
| **counter-fixed-504x800-mb1600-bc-seed1** | **BC** | **0.1** | **当前训练** |

用户最新要求是“停停停，用 BC 开始”。后续续训必须使用当前 BC 运行，不能误恢复 scratch 目录。BC 模型文件仍然存在，与是否在某次运行加载是两回事。

随机初始化那次初始贪心评估中，左侧 15 局零交付、零切菜完成，与等待对照一样；它不能作为 BC 权重退化的证据。当前 BC 运行确有后续回退，应以上面的独立记录分析。

## 环境与队友

任务为 V3 PartitionKitchen 的 `counter` 共享台面地图，13×7，中央墙隔成左一人、右两人。墙上五格可暂存物品，无需同拍同步交接。左右各有锅和加工设施，规则队友 `parallel-v2` 同时管理两口锅和订单。

| 菜谱 | 原料与加工 | 交付侧 | 时限 | 团队分数 |
|---|---|---|---:|---:|
| 柿子鸡蛋 | 切好柿子＋鸡蛋，入锅 | 右 | 80 | 20 |
| 煮马铃薯 | 一份切好马铃薯，入锅 | 左 | 60 | 10 |
| 马铃薯柿子牛肉 | 切好马铃薯＋切好柿子＋牛肉，入锅 | 左 | 100 | 50 |

奖励向量按菜谱 ID 为 `(20, 10, 50)`，超时不扣分。三单并发，随机抽样允许重复。柿子、马铃薯来源在左，鸡蛋、牛肉来源在右。切菜连续两步、看锅连续四步，放下/取走另占步；未完成工序漏拍进度清零。

每局仅一个厨师由模型控制，另外两名是规则队友。三种选单风格为队首优先、最早到期优先、原料最少优先；风格只影响选单，并非厨师编号。岗位按随机出生后的实际左右位置绑定。

| 阶段 | 左侧学习厨房 | 右主厨学习厨房 | 右传菜学习厨房 | 目标学习率 |
|---|---:|---:|---:|---:|
| left | 360 | 72 | 72 | 0.0003 |
| right_cook | 72 | 360 | 72 | 0.0003 |
| right_courier | 72 | 72 | 360 | 0.0003 |
| joint | 168 | 168 | 168 | 0.0001 |

三个岗位共享同一网络。重点阶段仍含另外两个岗位的样本，并不是只训练一个岗位。每组五局固定岗位和风格，组内出生和订单继续随机。

## 网络、历史和损失

- 当前观测 153 维；两名队友的联合动作编码为 36 类。
- 历史为五局，每局最近 40 个已发生的有效步；不是每局 800 或 2000 步全部进入历史。
- 每步历史输入 189 维，Episode MLP 为 189→256→256，再做有效步平均；Period 编码有效局平均后经过 256→256→256。
- 当前观测与 256 维历史拼成 409 维；Actor/Critic 各为 409→256→256，分别输出六动作和一个价值。
- Partner-class logits 为 256→3，预测队友选单风格，辅助交叉熵系数 0.1；不能把分类准确率当成协作成功率。
- PPO 使用 Adam、gamma 0.99、GAE lambda 0.95、clip 0.2、entropy 0.03、value 系数 0.5、最大梯度范数 15。
- 训练奖励为原生得分加 `0.99 * Phi(next) - Phi(current)`。势能依据当前订单需要的原料、加工与交付进度，并限制有用数量；真实局末下一状态势能取零。过程奖励可能为负，且属于团队信号。
- 训练时随机采样动作，验证时使用 argmax。初期贪心评估零分不代表训练探索中每一步都无效。

## 一轮到底做什么

训练是同步 PPO：先完成 `collect()`，再调用 `optimize()`；厨房之间批量并行，采样与优化没有并发运行。

一轮采 `504×800=403,200` 个学习者样本，每厨房两局，共 1,008 局。一个 epoch 为 252 个 mini-batch，四个 epoch 最多 1,008 次梯度更新。目标 KL 为 0.02，触发限制时会提前停止该轮剩余梯度更新，所以日志可能少于 1,008 次。

每阶段 10 轮为 4,032,000 步，四阶段为 16,128,000 步。每两个更新轮做一次 45 局验证。每阶段重新 warmup，前 2% 预算优化器步从目标学习率的 10% 线性升到目标值；因此初期日志 lr 可低于配置目标值。实际优化器步数受 KL 限制影响。

阶段切换保留模型和 Adam，重建环境与历史；仅在完整五局历史组边界切换。800 步采样下十轮正好 20 局，满足边界条件。验证分数不控制阶段推进。

## 代码与文件入口

| 内容 | 文件 |
|---|---|
| 当前固定课程入口 | [fixed_curriculum_experiment.py](fixed_curriculum_experiment.py) |
| PPO 采样、优化、完整状态 | [train.py](train.py) |
| 模型/观测/历史 | [policy.py](policy.py)、[features.py](features.py)、[history.py](history.py) |
| 双锅规则队友 | [partners.py](partners.py)、[controllers.py](controllers.py) |
| 过程奖励 | [shaping.py](shaping.py) |
| 评估/精简指标 | [evaluation.py](evaluation.py)、[bc_experiment.py](bc_experiment.py) |
| BC 采集训练 | [pretrain.py](pretrain.py) |
| 面板/回放导出 | [tracking.py](tracking.py)、[replay_export.py](replay_export.py) |

旧 `curriculum_experiment.py` 有成绩门槛语义，仅用于历史复现；当前用 `fixed_curriculum_experiment.py`。

运行目录内：`experiment.json` 为不可随意改动的配置；`train.jsonl` 每轮完整诊断；`validation/NNN.json` 为全局第 N 轮验证；`policy_NNNNNN.pt` 供评估；`latest.pt` 含模型、Adam、环境、队友、历史、随机数、课程进度，供续训；`finished_<stage>.pt` 为阶段完成模型。`status.json` 是最近一次状态发布，可能在验证进行中落后于 train.jsonl；它不是进程存活证明。summary.json 也可能是中途保存，不代表已经训练结束。

## 查看、续训和面板

以下命令均在项目根目录用 PowerShell 执行。使用项目 Python `hucooked/.conda/python.exe`。

```powershell
$env:PYTHONPATH = "$(Resolve-Path hucooked/src);$(Resolve-Path replay_frontend/src)"
Get-Content reproduction/nest_rl/mlp_partition_v3/runs/counter-fixed-504x800-mb1600-bc-seed1/status.json
Get-Content reproduction/nest_rl/mlp_partition_v3/runs/counter-fixed-504x800-mb1600-bc-seed1/train.jsonl -Tail 1
Get-Content reproduction/nest_rl/mlp_partition_v3/runs/counter-fixed-504x800-mb1600-bc-seed1/stderr.log -Tail 20
```

进程已在运行时不要再次执行训练命令。确认原进程退出后，同目录同参数运行会从 latest.pt 恢复；若还没有完整检查点，则从 BC 初始状态重新开始。

```powershell
hucooked/.conda/python.exe -u -m reproduction.nest_rl.mlp_partition_v3.fixed_curriculum_experiment `
  --campaign counter-fixed-504x800-mb1600-bc-seed1 `
  --bc-policy reproduction/nest_rl/mlp_partition_v3/runs/counter-bc-shaped-seed1/bc/bc_best.pt `
  --num-envs 504 --horizon 400 --rollout-steps 800 --minibatch-size 1600 --epochs 4
```

不要加 `--from-scratch`。不要把单独 policy 文件当作完整断点。完整检查点通常在每两轮评估后和阶段边界保存，崩溃时可能需要重跑最后未保存的轮次。恢复会按检查点裁剪超前的训练日志。

面板已在 7864 运行。只有面板退出时才另开终端执行：

```powershell
hucooked/.conda/python.exe -m reproduction.nest_rl.mlp_partition_v3.tracking --project hucooked-v3-cooperation
```

Trackio SQLite 位于 `runs/trackio`。精简面板优先看：

- `eval/score_per_episode`：验证团队平均分；训练均分另见 `train/score_per_episode`。
- `contribution/left`、`right_cook`、`right_courier`：该岗位团队均分减同岗位等待基线。负数表示模型使队伍比等待更差。
- `delivery_rate/<recipe>`：交付数除以交付加超时数；局末未决单不纳入这个分母。
- `delivery_steps/<recipe>`：成功订单从到达到交付的平均耗时，包含排队，不是纯加工时间；没有交付时无此指标。
- `train/process_reward_per_episode`：过程奖励，不能当原生分数。

`reward_per_lane` 是每厨房一轮总奖励，当前涵盖两局，不能直接当每局分数。aux accuracy 是风格分类准确率，不代表出餐或协作能力。

## 接手后的工作顺序与边界

1. 先确认当前 BC 进程与检查点，避免误启动 scratch 或重复占用 GPU。
2. 不改变用户已经确定的固定轮数、学习率、采样、mini-batch；需要诊断时另建评估输出目录。
3. 对第 6、8 轮左侧策略做同开局回放，核查连续工序是否被打断、原料是否滞留、交接是否等待，以及动作是否退化为重复移动。再决定是否需要修改训练。
4. 同时报学习者相对等待队伍的贡献、各菜交付率和出餐耗时；右侧绝对团队分数不能单独作为模型学会配合的证据。
5. 本次交接仅更新文档，没有修改当前训练行为，没有发起 L2 或独立长测，也没有提交或推送 Git。

旧文档保留在 [HANDOFF_HISTORY_20261124.md](HANDOFF_HISTORY_20261124.md)，其中“最新”指当时的历史状态，不是本次配置。旧报告包含统一每单 20 分、串行队友及 63 并发等口径，比较时必须确认奖励、控制器、局长和验证种子。

交接时工作区有大量未提交/未跟踪内容，V3 目录仍未被 Git 跟踪，runs 被忽略。换机器要单独传递代码、BC 权重、当前运行检查点、示范缓存和 Trackio 数据库；只拉分支无法恢复本次实验。不要顺手清理或覆盖其他目录的改动。
