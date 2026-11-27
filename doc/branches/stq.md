> 归档自 [stq 分支原 README](https://github.com/rain-xhy/hucooked-project/blob/6e9d36c23f5689d101207fbdc32ba67b531e5027/README.md)。运行命令均以仓库根目录为工作目录。

# STQ：SharedKitchen 基线复现与 PPO 实验

本分支包含 **26 组实验**：LIPO、CoMeDi、HiPT 的 11 组任务适配复现，以及 MLP、CNN、Transformer 三种基座的 15 组 PPO / BC→PPO。零分实验同样保留。

- [运行方式、方法说明和实验范围](../../reproduction/stq/README.md)
- [最终主策略的统一测评结果](../../reproduction/stq/RESULTS.md)
- [26 组实验与最终 checkpoint 清单](../../reproduction/stq/experiments.json)
- [最终权重与完整训练记录下载](https://github.com/rain-xhy/hucooked-project/releases/tag/stq-shared-final-20261127)

代码、配置、示范数据、验证记录和最终结果由 Git 保存；checkpoint 与完整大体积训练日志由 Release 保存。本次交付不包含 MixedKitchen 的实验代码、权重或结果。

## 目录

```text
overcooked_env/            厨房环境子模块，de37e081
replay_frontend/           Unity 回放子模块，072387b6
reproduction/stq/
  baseline_common/        分开动作方法共用的环境接口、示范与运行工具
  joint_common/           联合动作方法共用的奖励、示范与启动配置
  lipo/ comedi/ hipt/      分开动作方法
  joint_lipo/             联合动作 LIPO
  joint_comedi/           联合动作 CoMeDi，包括两种奖励
  joint_hipt/             联合动作 HiPT
  joint_ppo/              三种基座、三个模型规模的 PPO 实验
  scripts/                最终权重统一测评
  results/                最终测评 JSON / CSV
  experiments.json        配置、最终权重路径、训练步数与 SHA-256
artifacts/                本地下载的 Release 包，不纳入 Git
```

下载权重包并解压到 `reproduction/stq/` 后，26 组权重恢复到清单指定的各自路径。

## 取得项目

```bash
git clone --branch stq --recurse-submodules https://github.com/rain-xhy/hucooked-project.git
cd hucooked-project
```

两子模块保持公共主仓的版本与结构，没有为本次整理修改环境规则或 Unity 内容。环境子模块本身仍包含上游提供的其他厨房实现。安装、单组训练、整套训练及测评命令见[实验说明](../../reproduction/stq/README.md)。

## 结果口径

汇总使用**最终主决策策略**，不使用训练过程中的最佳权重。LIPO 评估全部 8 个最终成员；CoMeDi 评估最终 CBR；HiPT 评估最终分层主策略；PPO 评估最终 actor。

统一使用 SharedKitchen 默认任务、300 步局长、贪心动作、种子 100001，每个主策略或种群成员 12 局；指标直接读取 `info["metrics"]`。训练期间的历史验证分数另有记录，与该统一测评区分。

三个目标 1 亿步的 PPO 实验按用户要求提前停止，最终完整 checkpoint 分别为 MLP **57,950,400** 步、CNN **56,472,000** 步、Transformer **48,715,200** 步。固定验证得分长期未突破的证据见[结果说明](../../reproduction/stq/RESULTS.md)。

所有实验均为单个训练种子；12 局测评不代表多个独立训练种子的统计结果。项目公共介绍见 [main 分支](https://github.com/rain-xhy/hucooked-project/tree/main)。
