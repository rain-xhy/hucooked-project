# NestRL 复现与对照实验

本目录围绕 [NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming](https://arxiv.org/abs/2602.17737v2) 整理论文复现与标准 Overcooked 环境对照实验。**当前训练链路已打通，但尚不能宣称完整复现论文结果。**

## 两条实验路线

| 目录 | 目标 | 当前状态 |
| --- | --- | --- |
| [orgin/](orgin/) | 论文原始复现：使用论文作者改进的胡闹厨房环境和论文的嵌套训练方法训练、评估 L1/L2 策略。 | 已有训练、评估和性能工具；当前训练配置与论文仍有差异，详见下表。 |
| [cnn_standard/](cnn_standard/) | 标准 Overcooked 环境的空间观测与 CNN 实验，作为后续对照路线。 | 仅建立目录和说明，尚未实现或验证。 |

`adaptive-RL/` 是两条路线可参考的作者代码子模块，固定版本与配置见 [orgin 使用说明](orgin/README.md)。论文 PDF 和共享子模块保留在本目录；训练脚本、评估脚本和依赖清单位于 `orgin/`。本地生成的 `runs/`、`checkpoint-eval/`、依赖缓存及 Conda 环境仍保留在本目录，并被 Git 忽略。

## 复现进度与计划

勾选表示已有对应代码或历史验证，**不表示已经完成论文级复现**。历史结果和可执行命令见 [orgin/README.md](orgin/README.md)。

| 状态 | 工作项 | 现状 / 下一步 |
| --- | --- | --- |
| [x] | 获取论文与作者代码 | 保存论文 v2，固定作者 `adaptive-RL` 子模块版本。 |
| [x] | 作者环境与权重评估 | 已使用作者环境、配置和公开 PACE checkpoint 进行规则伙伴评估；不同伙伴模式的结果见详细说明。 |
| [x] | L1 与 L2 训练链路 | 已有从零训练 L1、使用冻结 L1 checkpoint 池训练 L2 的入口；短训练验证了链路，短跑不代表任务学成。 |
| [x] | 训练性能与记录工具 | 已有串行、子进程和 GPU 厨房相关实现，以及历史编码、profiling、学习率和可选 Trackio 记录工具。各模式的适用条件和结果需按配置验证。 |
| [x] | hucooked 特征适配探索 | 已有将本项目汤任务状态近似映射到 NestRL 73 维特征的适配器及检查脚本；它是迁移探索，不是论文原始环境复现。 |
| [ ] | 对齐论文训练设置 | 核对并实现附录 B.3 的 RNN 历史编码、网络与训练参数；当前脚本使用 MLP 历史编码，不能称为完整对齐。 |
| [ ] | 补齐论文评估条件 | 确认论文 L2 权重及完整自适应测试伙伴池的获取方式，建立与论文一致的训练、验证和测试划分。 |
| [ ] | 完成论文规模训练与复现实验 | 在对齐后的配置下完成多种子训练、评估与结果汇总，并和论文指标逐项比较；记录配置、种子和差异。 |
| [ ] | 实现标准环境 CNN | 在 `cnn_standard/` 接入标准环境的空间编码、CNN 策略及训练与评估入口，并明确地图尺寸和通道数的处理方式。 |
| [ ] | 设计可比对照实验 | 固定环境版本、地图、训练预算、种子、伙伴与指标，比较标准环境 CNN 路线和论文复现路线，说明环境和编码差异。 |

## 从哪里开始

1. 克隆后在仓库根目录初始化作者代码：

   ```powershell
   git submodule update --init -- reproduction/nest_rl/adaptive-RL
   ```

2. 阅读 [orgin/README.md](orgin/README.md)，按其中的环境准备和命令评估公开 checkpoint 或启动短训练。命令均从 `reproduction/nest_rl` 执行，输出写入该目录下的 `checkpoint-eval/` 或 `runs/`。
3. 标准环境 CNN 路线尚无可运行训练入口；后续工作见 [cnn_standard/README.md](cnn_standard/README.md)。

论文与代码：[arXiv v2](https://arxiv.org/abs/2602.17737v2) · [作者代码固定版本](https://github.com/upasana27/adaptive-RL/tree/33bb3222563735893ceb9258ebd3ea310583a148) · [论文 PDF](NestRL-2602.17737v2.pdf)
