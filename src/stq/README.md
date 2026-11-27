# SharedKitchen 多智能体实验复现

本目录包含 26 组 SharedKitchen 训练配置、训练代码和统一测评入口。厨房环境通过仓库根目录的 `overcooked_env@508bcbf` 提供。本文中的方法是面向三名厨师、SharedKitchen 观测与动作接口的适配实现，表格汇总本项目任务上的实验数据。

`baseline_common/` 与 `joint_common/` 提供环境、示范和运行工具；`lipo/`、`comedi/`、`hipt/` 保存分开动作版本，`joint_lipo/`、`joint_comedi/`、`joint_hipt/` 和 `joint_ppo/` 保存联合动作版本。每组结果留在各自独立的 `results/` 目录，精确路径和终态 checkpoint 由 `experiments.json` 列出。

## 实验范围

每个时间步，三名厨师各输出一个六选一动作。中心化策略对团队输入编码一次，再由三个角色动作头分别产生动作分布。分开动作策略则分别编码每名厨师自己的 866 维观察；其中包含 863 维公共全局状态和厨师角色标识，不是仅含私有局部信息的观察。

| 组别 | 配置数 | 方法和设置 |
| --- | ---: | --- |
| 论文方法：分开动作 BC+RL | 3 | LIPO、CoMeDi、HiPT 各一组；沿用 legacy 奖励配置 |
| 论文方法：联合动作 legacy | 6 | LIPO、CoMeDi、HiPT 各有纯 RL 与 BC+RL |
| CoMeDi 任务奖励对照 | 2 | 联合动作 CoMeDi，`ppo_task` 奖励下纯 RL 与 BC+RL |
| PPO small，2M 步 | 6 | MLP、CNN、Transformer，各有纯 RL 与 BC+RL |
| PPO small，20M 步 | 3 | MLP、CNN、Transformer，纯 RL |
| PPO large，20M 步 | 3 | MLP、CNN、Transformer，纯 RL |
| PPO million，目标 100M 步 | 3 | MLP、CNN、Transformer，纯 RL；按用户决定提前停止 |
| **合计** | **26** | **11 组论文方法对照，15 组 PPO** |

11 组论文方法中，9 组使用 legacy 奖励（3 组分开动作 BC+RL 与 6 组联合动作 LIPO/CoMeDi/HiPT）；另外 2 组是 CoMeDi `ppo_task` 奖励对照。15 组 PPO 均使用 `ppo_task`。除注明的 BC+RL 组外，纯 RL 不加载示范。BC+RL 使用随代码提供的 `baseline_common/demonstrations.npz`：32 局训练、8 局验证，共 9,600 个训练环境步和 2,400 个验证步；训练数据一半为规则教师轨迹，一半为带独立移动/等待扰动的纠偏轨迹。BC 为 64 个 epoch。预算只把 9,600 个示范环境步计入一次，不按成员数或重复训练轮数累加。

LIPO 保留 8 个成员，联合优化 self-play、cross-play 与 mutual-information 目标；CoMeDi 顺序生成 convention，结合 self-play、cross-play、mixed-play，再训练 CBR 主策略；HiPT 使用 16 个随机、训练中点与终态水平的伙伴策略，并保留高低层策略及影响奖励。联合动作 HiPT 每队只维护一次 CNN/编码器/LSTM 前向，同时保留三名厨师各自的 option、clock 和高低层回报。论文背景见 [LIPO](https://sites.google.com/view/iclr-lipo-2023)、[CoMeDi](https://arxiv.org/abs/2310.15414) 和 [HiPT](https://arxiv.org/abs/2305.16708)。

PPO 三种骨干均一次读取团队状态并输出三组动作 logits。MLP 使用 863 维公共状态；CNN 将 canonical spatial `[26,7,7]` 与公共状态融合；Transformer 编码空间 token 与公共状态 token。small、large、million 是不同网络宽度配置，参数量见下表。

下表为每个策略的 **actor 参数量**，不含 critic；数值与 `experiments.json` 中的 checkpoint 元数据一致。

| 方法/规模 | Actor 参数量 |
| --- | ---: |
| 分开动作 LIPO / 联合动作 LIPO | 60,742 / 62,162 |
| 分布式 CoMeDi / 联合动作 CoMeDi | 118,758 / 119,346 |
| 分布式 HiPT / 联合动作 HiPT | 392,410 / 393,326 |
| PPO small：MLP / CNN / Transformer | 129,426 / 132,482 / 126,914 |
| PPO large：MLP / CNN / Transformer | 291,602 / 354,034 / 310,450 |
| PPO million：MLP / CNN / Transformer | 974,738 / 1,064,402 / 993,490 |

## 安装

在支持 CUDA 的 Linux 环境中，从仓库根目录初始化环境和本目录依赖：

```bash
git submodule update --init overcooked_env
conda create --prefix ./src/stq/.conda python=3.10.22 pip -y
conda activate ./src/stq/.conda
python -m pip install -e ./overcooked_env
python -m pip install -r src/stq/requirements.txt
cd src/stq
```

当前代码使用 `overcooked_env@508bcbf`。其环境源码与归档实验所用的 `de37e081` 一致，历史权重仍可用于评估。训练需要可用的 NVIDIA 驱动和兼容的 PyTorch CUDA 运行环境，Python 依赖版本见 `requirements.txt`。

可在本目录运行 CPU 侧单元检查：

```bash
pytest -q
```

## 训练

例如，在物理 GPU 1 上运行一个新的 PPO small MLP 2M 纯 RL 实验：

```bash
RUN_OUT=outputs/ppo_mlp_small_rl_seed1
CUDA_VISIBLE_DEVICES=1 python -m joint_ppo.train \
  --mode rl --backbone mlp --model-size small --reward-profile ppo_task \
  --output "$RUN_OUT" --steps 2000000 --num-envs 16 --rollout 300 --workers 2 \
  --epochs 2 --minibatches 32 --lr 0.0001 --checkpoint-interval 50000 \
  --eval-interval 100000 --eval-episodes 12 --seed 1 \
  --device cuda:0 --physical-gpu 1 --memory-fraction 0.008
```

BC+RL 需要指定示范数据和另一条新的输出目录。可将 `BC_OUT` 设为 `outputs/ppo_mlp_small_bc_rl_seed1`，并在命令中使用 `--mode bc_rl --output "$BC_OUT" --demonstrations baseline_common/demonstrations.npz --bc-epochs 64`。两个输出路径都必须是新目录；不要重复使用已有实验目录。若选用其他 GPU，需令 `CUDA_VISIBLE_DEVICES` 与 `--physical-gpu` 使用同一个物理编号；`--device` 仍为暴露给进程的 `cuda:0`。

完整配置保存在下列 suite plan 中：

| Plan | 配置 | 启动入口 |
| --- | ---: | --- |
| `baseline_common/suite_plan.json` | 3 组分开动作 BC+RL | `baseline_common/launch_suite.py` |
| `joint_common/suite_plan.json` | 6 组联合动作 legacy + 6 组 PPO small 2M | `joint_common/launch_suite.py` |
| `joint_common/long_suite_plan.json` | 2 组 CoMeDi `ppo_task` + 3 组 PPO small 20M + 3 组 PPO large 20M | `joint_common/launch_suite.py` |
| `joint_common/million_suite_plan.json` | 3 组 PPO million 100M 目标 | `joint_common/launch_suite.py` |

计划中的 `output` 是归档实验原路径。复跑时先复制对应 JSON plan，将每个 job 的 `output` 改为新的、互不相同的目录；不要直接把复跑写入归档结果目录。以下示例展示复制和启动方式，运行前需编辑副本里的所有 `output`：

```bash
mkdir -p outputs/plans
cp baseline_common/suite_plan.json outputs/plans/baseline-fresh.json
cp joint_common/suite_plan.json outputs/plans/joint-fresh.json
cp joint_common/long_suite_plan.json outputs/plans/long-fresh.json
cp joint_common/million_suite_plan.json outputs/plans/million-fresh.json
# 将上述四份文件中每个 jobs[].output 改为新的、不重叠的输出目录
python baseline_common/launch_suite.py --plan outputs/plans/baseline-fresh.json
python joint_common/launch_suite.py --plan outputs/plans/joint-fresh.json
python joint_common/launch_suite.py --plan outputs/plans/long-fresh.json
python joint_common/launch_suite.py --plan outputs/plans/million-fresh.json
```

单任务可直接运行对应的 `python -m <method>.train`，并传入新的 `--output`。纯 RL 不传 `--demonstrations`；BC+RL 需要传入上述数据集路径。完整参数可通过对应入口的 `--help` 查看。

## 检查点与最终测评

所有 26 组均按**终态主策略**测评，不按 `best.pt` 挑选峰值。LIPO 测完每组最终种群的全部 8 个成员并等权汇总；CoMeDi 取最终 CBR 主策略；HiPT 取最终分层主策略；PPO 取最终 actor。三组 million 训练均按用户决定在 100M 目标之前停止，归档终态分别为 MLP `57,950,400` 步、CNN `56,472,000` 步、Transformer `48,715,200` 步，不从这些检查点续训。

三组 million 训练由用户决定停止，不是由训练评估自动终止。固定验证种子 120001 下，MLP 在第 5,001,600 步首次达到已见最高团队分 320，CNN 在第 2,400,000 步首次达到 320，Transformer 在第 5,702,400 步首次达到 380。从首次达到该分数到终态 checkpoint 的训练步数分别为 52,948,800、54,072,000、43,012,800，之后均未超过此前最高分。这里仅概述团队分；其他协作指标和策略行为可能仍有变化。

PPO 训练期间的原生验证记录与三条 million 训练的分数平台期另行汇总在 [`results/ppo_training_evaluations.csv`](results/ppo_training_evaluations.csv) 和 [`results/ppo_plateau_summary.json`](results/ppo_plateau_summary.json)。解压 Release 的训练记录包后，可从本目录重建：

```bash
python scripts/summarize_ppo_training.py
```

统一原生评估使用 12 局、种子 100001、300 步上限和 greedy 动作选择。3 个 LIPO 配置均测齐 8 个成员，其他 23 个配置各测一个最终主策略，共 564 局。分数和协作统计直接取环境 `info["metrics"]`；零分照实保留。每组训练种子为 1，这是单训练种子对照，12 局内的波动不代表跨训练种子的统计区间。评估清单和检查点路径见 [`experiments.json`](experiments.json)，汇总见 [`RESULTS.md`](RESULTS.md)、[`results/final_evaluation.json`](results/final_evaluation.json) 与 [`results/final_metrics.csv`](results/final_metrics.csv)。

从仓库根目录的 `src/stq/` 运行统一评估：

```bash
python scripts/evaluate_final_suite.py \
  --manifest experiments.json --gpus 1 --max-parallel 4 --episodes 12 --seed 100001 \
  --output results/final_evaluation.json
python scripts/summarize_final_results.py
```

## 下载权重和训练记录

普通 Git 内容包含训练源码、配置、实验清单与结果摘要。完整模型 checkpoint 和原始非 checkpoint 训练记录作为 GitHub Release 附件提供，不放入 Git 历史；训练记录包包含各组原始结果目录和 launcher 日志，不重复打包源码或评测脚本。附件文件名、字节数、SHA-256 与解压根目录以 [`release_assets.json`](release_assets.json) 为准；两包解压后保留实验原有目录结构。

```bash
mkdir -p ../../artifacts
gh release download stq-shared-final-20261127 \
  --repo rain-xhy/hucooked-project --pattern '*.tar.gz' --dir ../../artifacts
tar -xzf ../../artifacts/stq-shared-final-checkpoints.tar.gz -C .
tar -xzf ../../artifacts/stq-shared-training-records.tar.gz -C .
```

Release：[stq-shared-final-20261127](https://github.com/rain-xhy/hucooked-project/releases/tag/stq-shared-final-20261127)。
