> 本页记录 `orgin/` 的既有实现和历史实验。训练与评估结果文件保存在上一级 `runs/` 和 `checkpoint-eval/`；命令以 `reproduction/nest_rl` 为当前目录。当前实现与论文附录仍有差异，见[复现进度](../README.md)。

# NestRL 的 Overcooked 复现

Upasana Biswas、Durgesh Kalwar、Subbarao Kambhampati 和 Sarath Sreedharan，*NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming*。本目录收录未经修改的 arXiv v2 PDF，下载状态为 `open_access_downloaded`。论文采用 CC BY 4.0 许可。[arXiv 记录](https://arxiv.org/abs/2602.17737v2) · [PDF 来源](https://arxiv.org/pdf/2602.17737v2) · [许可](https://creativecommons.org/licenses/by/4.0/)

PDF 的 SHA-256 为 `c81b9eec95d79929b2516d18ecf0c2ec95cb9ee3deff1837a8be51f187819a7b`。arXiv 主站和官方镜像下载的文件逐字节一致。

`adaptive-RL` 子模块固定到作者公开仓库的 `33bb3222563735893ceb9258ebd3ea310583a148`。实验使用作者仓库内置的 Overcooked 环境及 `fc_small_test.yaml`，方便与其公开 checkpoint 对接。[作者代码](https://github.com/upasana27/adaptive-RL/tree/33bb3222563735893ceb9258ebd3ea310583a148)

在本项目根目录初始化源码后，进入复现目录创建 Windows CPU 环境。

```powershell
git submodule update --init -- reproduction/nest_rl/adaptive-RL
Set-Location reproduction/nest_rl
conda create --prefix .conda python=3.10 pip -y
conda run --prefix .conda python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
conda run --prefix .conda python -m pip install -r orgin/requirements.txt
conda run --prefix .conda python -m pip install --no-deps --ignore-requires-python pettingzoo==1.9.0 stable-baselines3==1.7.0
```

作者旧版 Gym 接口在 Windows 上使用 Gym 0.23.1。PettingZoo 1.9.0 的元数据限制 Python 版本，Stable-Baselines3 1.7.0 声明依赖 Gym 0.21。上述安装命令为 checkpoint 评估选用兼容版本组合。

## PACE 权重评估

用作者的 `ppo_pace_seed1` 权重与指定食谱的固定规则伙伴搭档，默认随机种子为 1，运行 10 轮，每轮 5 局。每轮局数与 checkpoint 的历史窗口一致。

```powershell
conda run --no-capture-output --prefix .conda python orgin/evaluate_pace.py --recipe lettuce-onion
conda run --no-capture-output --prefix .conda python orgin/evaluate_pace.py --recipe lettuce-onion --partner-mode minimum
```

原评估入口从 `environment/overcooked/policy.py` 取 `minimum` 伙伴。这种伙伴会取食材、放到隔板，并在成品可用时尝试送餐，但不切菜或装盘。作者的 L1 规则池在 `environment/overcooked/policy_server.py` 选用 `full`。保持 checkpoint、地图、初始随机种子和轮数一致，替换伙伴得到下列结果。

| 伙伴模式 | 食谱 | 完成局数 | 成功率 |
| --- | --- | ---: | ---: |
| `minimum` | Lettuce–Onion | 0/50 | 0% |
| `full` | Lettuce–Onion | 47/50 | 94% |
| `full` | Tomato–Carrot | 44/50 | 88% |
| `full` | Potato–Broccoli | 46/50 | 92% |

评估入口默认选择 `full`。它把指标字典从作者评估函数的两个返回值中取出，供原 CLI 汇总。逐局监控数据及结果 pickle 按权重、种子、伙伴模式和食谱存入 `checkpoint-eval/`。

## 从零训练 L1 和 L2

下面的命令以默认短训练设置启动两次右侧 L1 和一次左侧 L2。每个 L1 与三种全功能规则伙伴训练。L2 的训练池使用种子 1 的 L1，评估池使用种子 2 的 L1。每次运行使用一个独立的 `--run-name`，权重和启动配置写入 `runs/<run-name>/`。

```powershell
conda run --no-capture-output --prefix .conda python orgin/train_nestrl.py l1 --player 1 --seed 1 --run-name l1-right-s1
conda run --no-capture-output --prefix .conda python orgin/train_nestrl.py l1 --player 1 --seed 2 --run-name l1-right-s2
conda run --no-capture-output --prefix .conda python orgin/train_nestrl.py l2 --player 0 --seed 3 --run-name l2-left-s3 --partners runs/l1-right-s1/ppo/latest.pt --eval-partners runs/l1-right-s2/ppo/latest.pt
```

| 短训练 | 交互步数 | PPO 更新 | 末次评估完成局数 |
| --- | ---: | ---: | ---: |
| 右侧 L1，种子 1 | 1,200 | 2 | 0/15 |
| 右侧 L1，种子 2 | 1,200 | 2 | 0/15 |
| 左侧 L2，种子 3 | 400 | 2 | 0/5 |

公开训练入口默认只接规则伙伴池。训练脚本复用原来的 PPO、环境及历史缓冲区，在本地接入 L1 checkpoint 池，冻结 L1 参数，并在每步写入 L1 的观测和对手动作。L1 在每个五局回合中保留上下文，换回合清空。L2 评估中有 4/5 局在开局时读到了前面各局的历史。CPU 适配还处理了训练入口重复压缩动作维度的问题，以及规则伙伴无法向目标移动时的等待动作。

短训练用于验证链路。它沿用公开 checkpoint 配置中的 MLP 历史编码，论文附录 B.3 描述的是 RNN 历史编码。论文所用的 L2 权重及完整的自适应测试伙伴池没有出现在当前公开 checkout 中。可通过 `--rollout-steps`、`--updates`、`--processes` 调整本地训练规模。交互步数为三者乘积，厨房数至少等于训练搭档数，搭档按厨房编号轮流分配。 [论文附录 B.3](https://arxiv.org/html/2602.17737v2#A2.SS3)

## 并行厨房与性能测量

训练默认使用 `--env-mode serial`。至少配置两个厨房时，可用 `--env-mode subprocess` 为每个厨房启动独立进程，或指定 `--workers` 让多个厨房共享一个进程。工作进程数必须整除厨房数。每个 L2 厨房在 CPU 上加载自己的冻结 L1 搭档并维护历史，评估使用串行厨房。使用 CUDA 版 PyTorch 时，可加 `--device cuda` 在 GPU 上运行主策略和 PPO。

`--processes 3 --rollout-steps 400 --updates 5 --profile` 会采集 6,000 个训练环境步，并在运行目录写入 `profile.json`。其中主进程计时为墙钟秒，厨房计时按环境分别记录。多进程下各厨房的计时有重叠。

CUDA 策略采样默认将 MLP 历史编码整批执行。PPO 反向传播使用作者的编码路径。可用 `--history-encoder original` 对照原版。24 个厨房的 6,000 步短跑使用 `--device cuda --env-mode subprocess --processes 24 --rollout-steps 50 --updates 5`，每轮 PPO 收集 1,200 条交互样本。

`--ppo-mini-batch-size` 指定每次优化器更新使用的样本数。128 厨房短跑使用 `--device cuda --env-mode subprocess --processes 128 --workers 16 --rollout-steps 48 --updates 1 --ppo-mini-batch-size 512 --profile`。单轮采集 6,144 条样本，每遍训练恰好分成 12 个小批次。

## 学习率与 Trackio

`--learning-rate` 默认是 `0.001`。安装 Trackio 后，可加 `--trackio-project <name>` 将 PPO 指标和评估结果按环境步数记录到 `runs/trackio/<name>.db`。`--eval-every-updates` 控制评估间隔，须整除 `--updates`。`--eval-rounds` 指定每次评估对每个搭档重复的五局回合数。各回合使用固定评估种子，并在评估结束后恢复训练随机数状态。

`--lr-warmup-updates` 默认为 0。启用后，前几轮 PPO 按实际 Adam 更新次数将学习率从目标值的 10% 线性升至目标值。Trackio 每轮记录起始和结束学习率。

`--trackio-sample-every` 可在长 rollout 内按向量环境步记录已结束局的奖励、成功率和局数。PPO 损失继续按整轮更新汇总。
