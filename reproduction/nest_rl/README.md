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
conda run --prefix .conda python -m pip install -r requirements.txt
conda run --prefix .conda python -m pip install --no-deps --ignore-requires-python pettingzoo==1.9.0 stable-baselines3==1.7.0
```

作者旧版 Gym 接口在 Windows 上使用 Gym 0.23.1。PettingZoo 1.9.0 的元数据限制 Python 版本，Stable-Baselines3 1.7.0 声明依赖 Gym 0.21。上述安装命令为 checkpoint 评估选用兼容版本组合。

用作者的 `ppo_pace_seed1` 权重与固定的 Lettuce–Onion 规则伙伴搭档，随机种子为 1，运行 10 轮评估，每轮 5 局。每轮局数与该 checkpoint 的 5 局历史窗口一致。

```powershell
conda run --no-capture-output --prefix .conda python evaluate_pace.py
```

运行入口把指标字典从作者评估函数的两个返回值中取出，供原 CLI 汇总。日志及结果 pickle 位于 `checkpoint-eval/`。

Windows CPU 实测 50 局，完成 0 局，平均单局奖励 −6.82。作者的公开评估入口在此配置下选用一个固定规则伙伴，结果仅适用于这个 checkpoint 和配对。

公开仓库包含 PACE checkpoint。论文中嵌套训练的 L2 checkpoint 路径可在 `adaptive-RL/webapp/models.py` 找到，但该路径的权重文件及完整的自适应测试伙伴池未随此版本发布。
