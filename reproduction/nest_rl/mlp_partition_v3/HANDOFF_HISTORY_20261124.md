# V3 共享台面厨房 · L1 训练交接

## 双锅并行与分菜谱奖励更新

环境默认按煮马铃薯 10 分、柿子鸡蛋 20 分、马铃薯柿子牛肉 50 分计分。默认配置的 `recipe_rewards` 按菜谱编号保存为 `(20, 10, 50)`。新控制器 `parallel-v2` 同时维护两口锅的订单，支持两侧加工任意菜和成品跨墙交付；单目标控制器保存为 `serial-v1`。老训练状态按原控制器及每单 20 分恢复。

这次只重新评估已有 L1 权重。比较目录为 [parallel-v2-comparison](runs/parallel-v2-comparison/)，新旧规则队友都使用新分值、400 步局长和相同测试开局。回放在该目录的 `replays` 中，按控制器、队伍、岗位和风格分组。每帧保存 `state.native_state`，包含原始 V3 物品、工序和订单。诊断目录 `scenarios/dual-pot-cross-delivery` 从左右锅分别装好柿子鸡蛋和马铃薯的人工状态开始，验证同拍看锅及成品跨墙交付。

运行入口为 `python -m reproduction.nest_rl.mlp_partition_v3.compare_partners`，项目环境和 PYTHONPATH 见下文。Trackio 中筛选 `parallel-v2-comparison` 前缀可看到新结果。相比旧队友，新队友可以在学习者等待时自行承担更多任务，因此评价现有 L1 时要使用本次等待对照。

新旧控制器各跑完五组 450 局，共 4,500 局。两列分数均按新奖励计算，旧 L1 权重直接加载。

| 队伍 | 串行队友均分/局 | 并行队友均分/局 |
|---|---:|---:|
| 全脚本 | 261.80 | 312.00 |
| 学习者等待 | 75.84 | 178.96 |
| L1 seed 1，第 60 轮 | 127.53 | 132.00 |
| L1 seed 2，第 50 轮 | 131.91 | 116.49 |
| L1 seed 3，第 65 轮 | 131.84 | 115.09 |

全脚本均分提高 19.2%，每局交付从 11.17 单提高到 11.82 单。并行全脚本累计 42,210 个环境步双锅均有食材，3,264 步双锅同拍推进加工或取成品，7,776 次实际跨墙取物。三个旧 L1 都低于新队友的等待基线，不能将新脚本能力的提升视为旧 L1 的提升。已有独立测试支持的结论是并行脚本可以工作，旧 L1 需要在新队友条件下重新检验或训练。[完整比较结果](runs/parallel-v2-comparison/summary.json)

共导出 90 份配对评估回放和 9 份人工双锅场景回放，均为 400 步、401 帧。人工场景中九组都完成了最初两张异侧加工订单，其中三组记录到同拍看锅。已用项目 Unity 回放器加载并截图检查。[场景验收](runs/parallel-v2-comparison/scenarios/dual-pot-cross-delivery/validation.json) · [实际渲染截图](runs/parallel-v2-comparison/scenarios/dual-pot-cross-delivery/preview.png)

## 原 L1 实验记录

更新于 2026-11-24。主仓库分支为 `wjp`，正式实验名为 `counter-400-lr001`。三个训练种子和独立测试均已完成。**目前可以用作 L2 队友候选的是右侧传菜岗位。左厨在三个种子的独立测试里全部零交付，右侧主厨的贡献接近等待对照。** 下一步优先查左厨从走到原料源到完成第一次切菜的过程。L2 和过程奖励对照都还没有开训。[正式实验汇总](runs/counter-400-lr001/summary.json)

## 先找到代码和权重

本目录包含策略、训练器和评估器。环境规则在主仓库的 `hucooked` 子模块中。交接时本目录仍是主仓库的未跟踪目录，`runs/` 也被本目录的 Git 忽略规则排除。换机器时需另行传递代码、权重、评估 JSON、回放和 Trackio 数据库，仅检出分支无法还原本次实验。

| 要找的内容 | 入口 |
|---|---|
| 153 维观测、历史编码、Actor/Critic 和队友分类头 | [特征定义](features.py) · [策略](policy.py) · [Torchview 结构图](model_graphs/v3_policy_5x40_256.svg) |
| 三种规则队友、PPO、五局历史回放 | [规则队友](partners.py) · [训练器](train.py) · [历史缓冲](history.py) |
| 训练编排、验证选模、独立测试和消融 | [实验入口](experiment.py) · [评估口径](evaluation.py) |
| seed 1 / 2 / 3 按验证集选中的策略权重 | [第 60 轮](runs/counter-400-lr001/formal-lr0.001-seed1/policy_000060.pt) · [第 50 轮](runs/counter-400-lr001/formal-lr0.001-seed2/policy_000050.pt) · [第 65 轮](runs/counter-400-lr001/formal-lr0.001-seed3/policy_000065.pt) |

评估或冻结队友时读取 `policy_*.pt` 的 `model` 字段。续训使用同一运行目录下的 `checkpoint_*.pt`，它还包含 Adam、环境、队友、历史和随机数状态。旧实验 `counter-v1` 使用 300 步局长，学习率筛选后曾以 0.0003 开始正式训练，随后被本次 400 步实验取代。

## 这一轮怎么训

共享台面地图 `counter`，左侧一人、右侧两人，中央台面可暂存物品。三单并发，每次正确交付团队得 20 分。每局一个学习者，另两人为规则队友。63 个厨房均分给三个学习岗位和三种选单风格，每种组合 7 个。风格分别优先考虑队首、最早到期、原料最少的订单。厨师编号随出生位置变化。

模型使用当前厨师视角的 153 维观测，历史每步再拼接两名队友的联合动作（36 类）。最多保留五局、每局最近 40 个有效步；隐藏层宽 256。Actor 输出六个动作，Critic 输出状态价值，辅助分类头预测三种队友风格。

正式训练每局 400 步，每个厨房每轮采 800 步，共 50,400 条学习者样本。mini-batch 900，4 个 epoch，每轮更新 224 次。目标学习率 0.001，前 2% 优化器步从 0.0001 线性预热。三个种子各训 80 轮，即每个种子 4,032,000 个环境步。每 5 轮做 45 局验证；用验证成绩选检查点，再用独立种子做 450 局贪心评估，每个岗位 150 局。指标按同一批出生位置和订单抽样序列与等待、全脚本、空历史对照配对。[配置](runs/counter-400-lr001/config.json) · [PPO 参数和运行说明](README.md)

## 如何读结果

| 独立测试 | seed 1 | seed 2 | seed 3 | 学习者等待 | 全脚本 |
|---|---:|---:|---:|---:|---:|
| 三道菜宏平均交付率 | 34.34% | 34.89% | 34.97% | 23.96% | 64.13% |
| 团队平均分/局 | 109.11 | 110.76 | 111.02 | 73.38 | 223.47 |
| 学习者在左厨时的团队交付数 | 0 | 0 | 0 | 0 | 1,676 |
| 学习者在右侧主厨时的团队交付数 | 889 | 900 | 904 | 897 | 1,676 |
| 学习者在右侧传菜时的团队交付数 | 1,566 | 1,592 | 1,594 | 754 | 1,676 |

宏平均交付率先按菜谱计算“交付 ÷（交付＋过期）”，再等权平均；局末未决单单列。每局均分用团队交付总数乘 20、除以局数。面板上的训练指标 `reward_per_lane` 涵盖每个厨房一轮的两局，因此要除以 2 才得到训练每局均分。清空历史后三个种子的宏交付率分别为 34.75%、34.98%、34.84%，没有一致的下降趋势；目前没有证据说明五局历史提高了交付率。[分岗位原始记录](runs/counter-400-lr001/test-s1-policy.json) · [等待对照](runs/counter-400-lr001/test-wait.json) · [全脚本对照](runs/counter-400-lr001/test-script.json)

## 回放与故障定位

项目根目录下运行以下 PowerShell 命令即可用 Unity 回放器打开左厨失败案例。同目录有三个岗位各自的 L1、全脚本、等待轨迹，共九份。它们用 seed 1 第 60 轮策略、队首先做的规则队友和测试开局 20000000，从空历史开始，每份 400 步、401 帧。

```powershell
$replays = 'reproduction/nest_rl/mlp_partition_v3/runs/counter-400-lr001/unity-replays'
& .\replay_frontend\release\BetterReplay-Windows-x64-v0.2.1\OvercookedReplay.exe --replay (Resolve-Path "$replays/L1-left-chef1.ocrep.json").Path
```

先看 [L1 左厨](runs/counter-400-lr001/unity-replays/L1-left-chef1.ocrep.json)，再看 [同开局脚本左厨](runs/counter-400-lr001/unity-replays/script-left-chef1.ocrep.json)。两份都是厨师 1 在左侧。Unity 用洋葱代示土豆和牛肉，用盘子代示蛋；辨认原料和加工进度可查看帧里的 `state.v3` 或打开[同局的 V3 标注回放](runs/counter-400-lr001/L1-replay.html)。

- **左厨。** 三个种子的左厨测试分别只记录到 2、3、2 次团队切菜完成，均没有煮熟一道菜。示例回放里左厨走过 15 格，只交互 3 次，全程空手。连不需要跨墙的煮马铃薯也没有做成，先查取料与备料。
- **右侧主厨。** 三个种子的柿子鸡蛋交付都是 0。示例回放里该厨师按了 72 次交互，仍全程空手。队友能独立完成其他订单，总分掩盖了这个问题。
- **右侧传菜。** 学习者等待时团队交付牛肉菜 0 单，换成 L1 后分别交付 354、382、372 单。这个岗位值得保留为对照。上述工序统计都是团队事件，逐厨动作次数仅来自该示例局。

## 后续实验顺序

1. **先定位第一步失败在哪里。** 用同一检查点和同一测试开局比较贪心与按概率采样。只对学习者本人统计到达原料源旁、面向原料源、成功取料、放上菜板、完成切菜各环节的次数。同时记录无效交互面前的地块。这样可以区分“采样也拿不到料”和“训练时偶尔会做、贪心评估时不做”。
2. **再验证奖励假设。** 当前 V3 只有交付奖励。作者 NestRL 的这次 Overcooked 运行对首次出现的相关持物状态、菜谱子目标进展和完成菜谱分别使用 0.5、1.0、10 的系数，另有每步 0.2 的成本。本地旧三人交接也把塑形奖励加进了 PPO 回报。V3 的左厨难以得到早期反馈、右侧主厨可随队友获分，是合理解释，尚未通过对照确定因果。建议固定网络、种子、队友和步数，只改变训练奖励；评估统一用原生交付分与分岗等待对照。过程奖励按实际状态变化发放，避免重复拿放或空交互刷分。[作者配置](../adaptive-RL/environment/overcooked/config/forced_coord_1_to_1_full.yaml) · [奖励实现](../adaptive-RL/environment/overcooked/gym_cooking/environment/cooking_zoo.py) · [本地旧交接训练](../mlp_standard/handoff_kitchen.py)
3. **最后决定 L2 队友池。** 先冻结两名 L1，在同一组开局里分别搭配脚本第三人和等待第三人，轮换三个岗位。先确认哪些组合能稳定完成跨墙菜，再训练一名 L2 配两名冻结队友。候选划分为 seed 1、2 训练池，seed 3 整个种子留作测试。每名冻结厨师要有自己视角的历史。队友身份标签、L1 权重初始化对照属于 L2 设计，当前三分类头的语义仍是规则队友风格。

## 本地面板和验证

项目环境为 `hucooked/.conda`。从主仓库根目录运行；面板已在运行时直接打开[本地 Trackio](http://127.0.0.1:7864/?project=hucooked-v3-counter-ppo)。项目名为 `hucooked-v3-counter-ppo`，选择前缀 `counter-400-lr001` 的记录。

```powershell
$env:PYTHONPATH = "$(Resolve-Path hucooked/src);$(Resolve-Path replay_frontend/src)"
.\hucooked\.conda\python.exe -m unittest discover -s reproduction/nest_rl/mlp_partition_v3 -t . -v
.\hucooked\.conda\python.exe -m reproduction.nest_rl.mlp_partition_v3.tracking
```
# 2026-11-24 补充：BC → 带过程奖励的 PPO

入口 `bc_experiment.py`，默认输出 `runs/counter-bc-shaped-seed1/`。新增 `pretrain.py` 生成并行脚本示范、训练 actor 与历史编码器，最佳模型为 `bc/bc_best.pt`；PPO 从该模型初始化，重建 Adam。新实验保留 seed 1、63 厨房、400 步局长、800 步采样和 0.001 学习率，80 轮预算。

`shaping.py` 定义 `order-potential-v1`。训练奖励为环境交付分数加 `0.99*Phi(next)-Phi(current)`，按当前订单需求封顶、终局势能归零。原生环境与评估分数没有过程奖励。旧训练状态默认关闭 shaping，可继续恢复。完整训练状态记录 shaping 版本与系数。

Trackio 新项目 `hucooked-v3-cooperation` 使用原来的本地 7864 端口。`tracking.py` 默认打开新项目，旧项目可通过 `--project hucooked-v3-counter-ppo` 打开。新面板保留 BC 两项指标与 PPO 最多十六项指标，核心是相对等待队伍的得分贡献、按岗位拆分的贡献和三道菜交付率。完整损失、分类、流程诊断仍写 JSON，不再全部推到面板。

验证种子：BC 30,000,000；PPO 10,000,000。PPO 每 5 轮进行 45 局验证，缓存同种子的 wait/script 基线。训练恢复复用已完成 BC，从最近完整 checkpoint 继续。当前补充说明实现与启动方式，实际进度和成绩以该 run 的 `bc/bc_metrics.json`、`ppo/train.jsonl`、`ppo/validation/` 为准。


## 分岗位重训入口

已新增 `curriculum_experiment.py`，独立运行目录 `runs/counter-curriculum-seed1`。从 BC 最佳权重重新开始，按左厨→右主厨→传菜→联合顺序，45/9/9 轮换到最后21/21/21，并保留BC示范损失、KL更新门槛和阶段成绩保留门槛。详细参数、晋级条件、恢复方法见 README 的“分岗位课程训练”。`status.json` 为实时进度；`summary.json` 与 `latest.pt` 对应保存点。阶段最多20轮、连续两次评估合格才晋级；未达标停止并写明原因。不能把尚未晋级的状态理解为已经完成四个岗位阶段。

2026-11-24 后续调整：新课程实验改用 `runs/counter-curriculum-lr0003-seed1`，重点阶段学习率 0.0003、联合微调 0.0001，从相同 BC 最佳权重重启；原课程记录保留。入口新增 `--focus-lr` 和 `--joint-lr`，其值写入实验配置；复现原配置需显式传入 0.001 和 0.0003。预算结束只通过一次验证时，停止原因明确记录连续通过次数不足，避免 `needs_attention` 配空原因。

随后用户确定使用 128 个并发厨房并保持总环境步数大致不变。当前默认实验为 `runs/counter-curriculum-128-lr0003-seed1`：重点岗位 92/18/18 轮换，联合 43/43/42，每阶段最多 10 轮；四阶段预算最多 4,096,000 步，比原 4,032,000 步增加约 1.59%。验证按 252,000 环境步阈值，在各阶段第 3/5/8/10 轮进行；晋级仍限完整五局历史组的第 5/10 轮。第 8 轮达到连续通过条件也不能立即切换，必须在第 10 轮重新验证，保存恢复遵守同一规则。旧 63 厨房运行恢复时指定 `--num-envs 63 --stage-updates 20`，并按其历史记录设置学习率。

最新试验配置为 252 厨房：`--campaign counter-curriculum-252-lr0003-seed1 --num-envs 252 --stage-updates 5`。入口允许每阶段最少 5 轮，默认参数仍保留 128 厨房/10 轮。252 的岗位分配为 180/36/36 轮换、84/84/84 联合；每阶段精确保持原 1,008,000 环境步预算。评估在第 2/3/4/5 轮，晋级只在第 5 轮历史组结束处发生。第 4 轮已连续通过后恢复仍不会提前晋级，第 5 轮重新验证才切换。CPU 回归覆盖这条边界。

用户最终确定每厨房每次只采 400 步（一完整局），因此正式命令改为 `--campaign counter-curriculum-252x400-lr0003-seed1 --num-envs 252 --rollout-steps 400 --stage-updates 10`。每阶段仍为 1,008,000 环境步，总上限仍为 4,032,000。每阶段第 3/5/8/10 轮评估，晋级仅限第 5/10 轮的完整五局历史组边界。岗位数、学习率、示范库与 45 局验证均不变；入口默认值未更改，使用显式参数区分这次配置。

后续新增每厨房采样 1000 步的试验：`--campaign counter-curriculum-252x1000-lr0003-seed1 --num-envs 252 --rollout-steps 1000 --stage-updates 4`。局长依旧 400，奇数轮结束保留 tick=200 的半局并继续下轮；每阶段预算与前述配置相同。每轮 252,000 环境步，所以每轮评估，晋级只允许第 2/4 轮完整五局历史组边界。runner 采用 `(update * rollout_steps) % (horizon * 5) == 0` 判断，不对 1000/400 做整数截断。阶段预算只需正整数且最终对齐历史组；CPU 测试覆盖四轮晋级/失败、tick=200 保存恢复与原配置兼容。

## 最新：固定四阶段、504×800

用户取消了按分数提前晋级和旧岗位退步停止，要求从 BC 权重完整重训。新入口 `fixed_curriculum_experiment.py` 与旧门槛入口分开；正式命令 `--campaign counter-fixed-504x800-lr0003-seed1`，默认 504 厨房、局长 400、采样 800，每阶段固定 10 轮，总计 40 轮、16,128,000 环境步。岗位分配 360/72/72 轮换，联合 168/168/168。重点学习率 0.0003、联合 0.0001，BC 0.1 与 KL 0.02 保留。

初始 BC 模型评估后，每阶段第 2/4/6/8/10 轮评估；评估不参与阶段控制。第 10 轮结束才切阶段，模型及 Adam 保留，环境和历史重置，每阶段重新 warmup。`latest.pt` 保存全部恢复状态，`finished_<stage>.pt` 记录固定阶段的完成模型。中途短跑保存不会标记完成；只有全局 40 轮结束才为 `complete`。CPU 回归覆盖高分不提前、低分不中止、四阶段40轮、Adam状态延续，以及第10轮存盘未切换的故障恢复不会重跑该阶段。

## 最新：跳过 BC，随机初始化的 504×2000 PPO

用户要求全部从头、跳过行为克隆。入口新增 `--from-scratch`，正式命令 `--campaign counter-fixed-504x2000-scratch-seed1 --num-envs 504 --rollout-steps 2000 --from-scratch`。局长仍 400，四阶段各固定 10 轮，预算合计 40,320,000 环境步。网络和 Adam 从 seed 1 新建；既不读取 BC artifact，也不调用示范采集/加载和挂载接口，BC 损失系数为 0。保留 shaping、辅助分类系数 0.1、KL 0.02、前三阶段学习率 0.0003/联合 0.0001。

初始随机网络保存在 `initial_random.pt`，以同一网络做 `validation-initial.json`；恢复时优先加载完整 checkpoint，不覆盖已学权重。旧 BC 模式的 manifest 结构及恢复方式保留。新目录缓存只允许复用等待和全脚本基线，不复制 BC 示范或 BC 初始评估。CPU 测试确认随机初始化与 seed 一致、示范接口完全未调用、随机权重 artifact 未被训练修改，以及恢复后沿用已训练参数。
