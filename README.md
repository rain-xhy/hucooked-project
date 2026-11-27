# HuCooked 多智能体协作实验

本分支研究三名厨师如何在有限步数内完成更多订单。项目从 HAHA（HA²）的分层策略出发：Manager 选择下一项工作，Worker 决定厨师如何走位和交互。在 Shared 厨房上，我们比较了从随机初始化训练、模仿规则教师、规划器辅助的价值学习，以及从原有技能中重新学习团队分工四条路线。

**看结果**可直接打开 [Shared 五版榜单](hucooked%20leaderboard/shared/README.md)。表中每个版本都有权重、训练设置、24 局逐步评估和 Unity 回放。**复评最新的分段高层 PPO** 可按下方命令在 CPU 上加载已提交的权重。**复训**则使用后面的独立入口。

## Shared 榜单怎样读

五版都在同一固定厨房中运行。每局 300 步，普通订单和协作订单每送出一单均得 20 分。协作订单需要两名厨师在对应位置同时出锅。官方评估使用相同的 24 个种子（84031–84054）；这些结果衡量该厨房上的表现，跨地图泛化尚待验证。

| 方法 | 执行时控制三名厨师的模型 | 平均出餐 | 平均得分 | 同步成功率 | 每局等待步数 | 订单完成率¹ | 平均交付步数¹ |
|---|---|---:|---:|---:|---:|---:|---:|
| [随机初始化纯 PPO](hucooked%20leaderboard/shared/multi-agent/runs/01-pure-ppo-380/README.md) | 分层 Manager 和 Worker，执行参数 69,394 | 19 | 380 | 21.74% | 45 | 90.48% | 29.47 |
| [420 规则策略 + BC](hucooked%20leaderboard/shared/multi-agent/runs/02-rule-bc-420/README.md) | 模仿规则分工与动作的 Manager 和 Worker，执行参数 69,394 | 21 | 420 | 18.03% | 50 | 91.30% | 27.71 |
| [440 周期规则策略 + BC](hucooked%20leaderboard/shared/multi-agent/runs/03-rule-bc-440/README.md) | 可观测周期相位的 Manager 和 Worker，执行参数 85,007 | 22 | 440 | 100% | 11 | 91.67% | 26.59 |
| [420 BC + 规划器引导的 Q 网络](hucooked%20leaderboard/shared/multi-agent/runs/04-planner-q-440/README.md) | 集中式联合动作网络，59,992 个参数，每步从 216 个动作组合中选一个 | 22 | 440 | 100% | 11 | 95.65% | 26.64 |
| [420 BC + 分段高层 PPO](hucooked%20leaderboard/shared/multi-agent/runs/05-segment-ppo-440/README.md) | 78 个可学习高层参数，配合 13 个工作片段和导航程序 | 22 | 440 | 50% | 12 | 91.67% | 25.73 |

¹ 两项均汇总全部 24 局。订单完成率 = 已交付订单总数 ÷ 已发布订单总数。平均交付步数 = 已交付订单从发布到送达的总步数 ÷ 已交付订单总数。[查看公式与逐版计数](hucooked%20leaderboard/shared/README.md#订单指标的计算口径)

**同步成功率**的分母是有效的协作出锅尝试步数。**等待步数**是协作菜具备出锅条件的步数减去成功出锅步数。最后一版每局 22 次尝试、11 次成功，共交付 22 单。两个指标分别反映交互时机和等待成本。

三版 440 的证据含义各不相同。周期 BC 直接模仿一份已知能得 440 的排程。离线 Q 使用 420 策略与规划器候选各自的前 90 步，其中一个候选已知能得 440，再用出餐奖励估计长期动作价值。分段 PPO 从 420 模型的一局行为提取工作片段，只用自己采样的出餐回报学习分配、先后和工位；片段边界和底层导航由程序定义。把 420 行为转换成初始片段策略时实测为 20 分，第 23 轮更新后选出的确定性策略在原生环境复评为 24 局 440。

## 代码和实验文件

| 目录或入口 | 内容 |
|---|---|
| [hucooked haha/haha](hucooked%20haha/haha/) | HAHA 分层训练、子任务与观测编码、张量化厨房、并行采样及 PPO 的可复用实现 |
| [hucooked leaderboard](hucooked%20leaderboard/) | 按 original、v1、v2、v3、shared 组织的环境适配、训练入口与运行产物 |
| [run.py](hucooked%20leaderboard/run.py) | 通用实验启动器。记录训练命令、环境修订和种子，调用对应版本的训练、统计与回放入口 |
| [v2/pure_rl.py](hucooked%20leaderboard/v2/pure_rl.py)、[v2/train.py](hucooked%20leaderboard/v2/train.py) | Shared 的纯 PPO 与规则 BC 复用的训练器。Shared 选择连续订单配置 |
| [shared](hucooked%20leaderboard/shared/) | 周期排程与走位、离线 Q、分段高层 PPO、CUDA 采样内核，以及分工和时序的历史对照代码 |
| [results.py](hucooked%20leaderboard/results.py)、[render_replay.py](hucooked%20leaderboard/render_replay.py) | 汇总原生评估，逐步验证回报后导出 Unity 回放 |
| [历史配对比较](hucooked%20leaderboard/comparisons/20261124-fresh.json) | original 至 v3 的多智能体与单智能体配规则队友实验结果 |
| `hucooked/`、`replay_frontend/` | 环境与回放器的 Git 子模块。榜单按环境提交号导出源码到本地 `.environment/` |

五个入榜目录使用相同结构：

- `README.md` 记录权重来源、训练轮数、环境步数、局数、学习率、耗时和计时范围。
- `train/best.pt` 是该方法的入榜权重。五种方法的参数格式分别由对应模型定义解释。
- `train/best-evaluation.json.gz` 是 24 局原生评估，含逐步动作、事件、奖励和订单。
- `scores.json` 是机器可读指标，`evidence/` 保存原始日志及该版所需的配置或规划输入。
- `replay/shared.ocrep.json` 可直接在 Unity 播放器中打开；`shared.trajectory.json` 是可重新渲染的轨迹。每份回放使用种子 84031，含 300 个动作步、301 帧。

## 准备 Shared 环境

从 `fgy` 分支克隆，在仓库根目录执行。以下示例使用 Bash 或 Zsh、Python 3.11 和独立虚拟环境。CUDA 训练还需要可用的 NVIDIA 驱动及与其匹配的 PyTorch；编译分段采样内核需要 CUDA Toolkit 和 `nvcc`。

```bash
git clone --branch fgy --recurse-submodules git@github.com:rain-xhy/hucooked-project.git
cd hucooked-project
python3.11 -m venv ../hucooked-venv
source ../hucooked-venv/bin/activate
python -m pip install -r "hucooked leaderboard/requirements.txt"
python "hucooked leaderboard/prepare_environments.py" shared

repo_root="$PWD"
board="$repo_root/hucooked leaderboard"
runs="$board/shared/multi-agent/runs"
export PYTHONPATH="$repo_root/.environment/shared/src:$repo_root/hucooked haha:$board${PYTHONPATH:+:$PYTHONPATH}"
work_dir="$(mktemp -d)"
```

Shared 固定使用环境提交 `de37e0818c2e76311e105ab8689e5162bd84133f`。准备脚本优先读取本地子模块已有对象，必要时从环境仓库获取对应提交。其他榜单环境可将准备命令中的 `shared` 换为 `original`、`v1`、`v2` 或 `v3`。各版本的提交号在 [run.py](hucooked%20leaderboard/run.py) 中维护。上面的 `work_dir` 是本次新输出目录，可自行换成持久保存实验的路径。

## 复评权重与查看回放

直接观看已归档结果时，在队友的 Unity 回放器中选择对应的 `shared.ocrep.json`。[最新分段高层 PPO 的回放](hucooked%20leaderboard/shared/multi-agent/runs/05-segment-ppo-440/replay/shared.ocrep.json)

下面的命令使用 CPU 加载分段高层 PPO 权重，在原生 Shared 环境运行 24 局，并将其中一局导出为新的 Unity 回放。依赖和环境变量沿用上一节。

```bash
python "$board/shared/autonomous_options.py" \
  --evaluate-policy "$runs/05-segment-ppo-440/train/best.pt" \
  --output "$work_dir/segment-eval"
python "$board/render_replay.py" shared \
  de37e0818c2e76311e105ab8689e5162bd84133f "$work_dir/segment-eval"
```

输出目录中的 `official-evaluation.json.gz` 用于逐局复核，`shared.ocrep.json` 用于播放。Unity 项目和已发布的播放器由队友维护。[回放器仓库](https://github.com/LynxPeng/Better-replay-frontend-of-overcook-ai)

## 分别复训五种方法

以下命令从仓库根目录执行，沿用前面的变量。不同训练器会在输出目录写入权重和日志。选中权重的更新次数与整次作业的预算可能不同，计数详见各版本训练记录。

**1．随机初始化纯 PPO。** 连续运行技能训练、联合微调、固定 Worker 的 Manager 微调。第二段读取第一段最佳权重，第三段读取第二段末权重。

```bash
python -m v2.pure_rl --device cpu --seed 488 --output "$work_dir/pure-base"
python -m v2.pure_rl --device cpu --seed 488 --updates 72 \
  --resume "$work_dir/pure-base" --output "$work_dir/pure-joint"
python -m v2.pure_rl --device cpu --seed 488 --updates 84 \
  --resume "$work_dir/pure-joint" --resume-last --throughput --manager-only \
  --output "$work_dir/pure-manager"
```

**2．420 规则策略 + BC。** 选用连续订单、提前备料和子目标约束。入榜权重来自首次 BC；原始训练器后续还会执行恢复模仿和 PPO，并记录选择阶段。

```bash
python -m v2.train --device cpu --seed 488 --mode multi --profile continuous \
  --early 3 --pipeline --goal-lock --updates 12 --test-seed-base 84031 \
  --test-episodes 24 --output "$work_dir/rule-bc-420"
```

**3．440 周期规则策略 + BC。** 使用归档的教师方案生成示范，`bc.pt` 是模仿结束时的模型。

```bash
python "$board/shared/train_schedule.py" \
  --plan "$runs/03-rule-bc-440/evidence/plan.json" --output "$work_dir/rule-bc-440"
```

**4．420 BC + 规划器引导的离线 Q。** 使用归档的三个候选方案和第二版的权重。此入口在 CUDA 上训练两种宽度，归档的是宽度 128 的开放动作评估版本。

```bash
python "$board/shared/prep_neural_q.py" \
  --checkpoint "$runs/02-rule-bc-420/train/best.pt" \
  --proposals "$runs/04-planner-q-440/evidence/proposals" \
  --output "$work_dir/planner-q"
```

**5．420 BC + 分段高层 PPO。** 在 Linux CUDA 主机上运行。下面的编译参数针对实验所用的 RTX 3090，其他 GPU 需要选择对应的编译架构。训练先从 420 权重提取片段，再用新交互的团队回报更新分工策略。

```bash
nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC \
  "$board/shared/autonomous_kitchen.cu" -o "$work_dir/kitchen.so"
python "$board/shared/autonomous_options.py" --method ppo --seed 491 \
  --library "$work_dir/kitchen.so" --checkpoint "$runs/02-rule-bc-420/train/best.pt" \
  --output "$work_dir/segment-ppo"
```

需要对照历史实验时，可在准备对应环境后通过 `run.py <版本> --mode multi-agent` 或 `--mode single-agent` 启动。用同一新 `--run-id` 完成两个模式后，`compare.py <run-id> --version <版本>` 会按种子配对比较。上述五版 Shared 榜单各自使用训练器和固定评估设置，训练数据来源、可执行的策略空间和计算预算均有差异。
