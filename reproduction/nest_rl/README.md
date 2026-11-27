# NestRL 复现与三人共享厨房协作训练

本目录围绕 [NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming](https://arxiv.org/abs/2602.17737v2) 开展多智能体协作复现，并在项目最新的**三人共享厨房环境（`SharedKitchen`，连续混合订单，7×7 网格，376 维观测，6 个原始动作）**上完成了纯 GPU 端到端训练与系统基准评测。

---

## 一、当前核心成果总览

### 1. 最终模型成绩（纯 0-BC、无历史编码器、`326k` 参数）

我们最终在 [mlp_shared/](mlp_shared/) 下基于**无历史编码器的轻量化 Ego-MLP 策略**（仅 `326,407` 参数，直接输出 6 个原始基础动作，不使用宏动作），在**零行为克隆（0-BC）**条件下，经 **L1 → L2 → L3** 三阶段 NestRL 嵌套训练，取得以下结果：

- **纯自博弈（3 个同模型贪心策略）**：**`21.00` 餐 / `420.00` 分**（普通 11 餐 + 联合 10 餐，同步成功率 `34.41%`，`0` 资源冲突），在原始动作空间上追平 `origin/hy` 分支带宏动作与学习开局分工的 `Ours`（`21.00 / 420.00`），超过 ACAC 基线（`20.68 / 413.59`）。
- **开局站位扰动（1 格邻域随机偏移，96 局）**：L3 达到 **`20.86` 餐 / `417.29` 分**（同步成功率 `34.34%`），高于 `origin/hy` 的 `Ours`（`20.49 / 409.79`）。
- **队友动作扰动鲁棒性（225 局，5% / 10% 队友随机动作）**：L2 在 5% 扰动下仍保持 **`322.49` 分（16.12 餐）**，10% 扰动下保持 **`258.40` 分（12.92 餐）**；L3 在无扰动下锁定最高上限 **`420.00` 分**。
- **GPU 训练吞吐**：关闭历史编码器 + `CUDAGraph` 向量化环境，训练吞吐达到 **`150k ~ 195k env steps/s`**，完整 200 轮（1.21 亿步）仅需约 12 分钟。

> **与 `origin/hy` 最新环境核查**：已核对 `origin/hy` 最新提交（`0cc6226` / `hucooked@508bcbf`），其环境核心为连续混合订单（`continuous_orders=True`、`OBS_DIM=376`、`MIXED_ORDER_OBSERVATION_SLOTS=8`、`joint_wait_steps`、`meals_per_100_steps`），与 `mlp_shared/env.py` 中的 `TorchSharedKitchen` 规则和指标定义保持一致。

---

### 2. 最新 Benchmark 结果

#### (1) 对齐 `origin/hy` 评估协议（96 局，3 策略纯贪心 `argmax`）

| 模型 / 方法 | 场景 | 局数 | 平均出餐数 | 平均团队得分 | 同步成功率 | 平均等待步数 | 冲突 (资/移) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NestRL L3 (`U025`，无历史 0-BC)** | 标准连续混合订单 | 96 | **21.00** | **420.00** | **34.41%** | 42.85 | 0.00 / 7.05 |
| **NestRL L3 (`U025`，无历史 0-BC)** | 开局站位扰动 (`jitter1`) | 96 | **20.86** | **417.29** | **34.34%** | 41.55 | 0.00 / 7.51 |
| **NestRL L2 (`U200`，无历史 0-BC)** | 标准连续混合订单 | 96 | 20.00 | 400.00 | 27.65% | 36.79 | 0.00 / 7.31 |
| **NestRL L2 (`U200`，无历史 0-BC)** | 开局站位扰动 (`jitter1`) | 96 | 19.96 | 399.17 | 28.17% | 37.47 | 0.00 / 7.07 |
| **NestRL L1 (`U200`，无历史 0-BC)** | 标准连续混合订单 | 96 | 3.41 | 68.12 | 42.21% | 133.21 | 0.00 / 36.45 |
| *参考：`origin/hy` Ours（宏动作+开局分工）* | 标准连续混合订单 | 96 | 21.00 | 420.00 | 30.05% | 25.51 | — |
| *参考：`origin/hy` Ours（宏动作+开局分工）* | 开局站位扰动 | 96 | 20.49 | 409.79 | 28.24% | 25.16 | — |
| *参考：`origin/hy` ACAC 基线* | 标准连续混合订单 | 128 | 20.68 | 413.59 | 25.90% | 29.08 | — |

> 注：在**全图任意地板随机出生点**（`random_floor`，96 局）这一训练期未开启随机出生的极端 OOD 场景下，L2 / L3 分别为 `167.92` / `165.83` 分（若需全图随机出生泛化，可开启 `--random-spawn-prob 0.5`，历史 DR 实验已验证可拉升至 `283.5` 分）。

#### (2) L1 / L2 / L3 × 队友动作扰动 9 配置 Benchmark（225 局，纯贪心 `argmax`，`shuffle_players=True`）

每局指定 1 个 ego 厨师（在 Server / Main / Side 间均分）保持 0% 噪声纯贪心，**其余 2 名队友按 `0% / 5% / 10%` 概率执行随机动作**：

**A. 全员神经网络自博弈（`all_policy`：3 个同阶段策略）**

| 模型 | 队友扰动 | Team Score | Meals | `j_succ` | `j_wait` | 冲突 (资/移) | 分角色 (`S / M / D`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1** (`U200`) | 0% | 72.53 | 3.63 | 37.5% | 144.4 | 0.00 / 27.11 | `66.9 / 75.2 / 75.5` |
| **L1** (`U200`) | 5% | 108.36 | 5.42 | 31.2% | 81.6 | 0.01 / 48.25 | `104.3 / 106.4 / 114.4` |
| **L1** (`U200`) | 10% | 95.29 | 4.76 | 41.7% | 43.1 | 0.03 / 55.84 | `97.6 / 90.4 / 97.9` |
| **L2** (`U200`) | 0% | 400.00 | 20.00 | 27.7% | 37.1 | 0.00 / 7.19 | `400.0 / 400.0 / 400.0` |
| **L2** (`U200`) | **5%** | **322.49** | **16.12** | 26.1% | 48.5 | 0.08 / 20.46 | `342.4 / 321.9 / 303.2` |
| **L2** (`U200`) | **10%** | **258.40** | **12.92** | 24.3% | 57.9 | 0.10 / 34.32 | `287.5 / 255.5 / 232.3` |
| **L3** (`U025`) | **0%** | **420.00** | **21.00** | **34.4%** | **42.5** | 0.00 / 7.22 | **`420.0 / 420.0 / 420.0`** |
| **L3** (`U025`) | 5% | 304.27 | 15.21 | 34.5% | 45.3 | 0.15 / 21.09 | `334.7 / 296.5 / 281.6` |
| **L3** (`U025`) | 10% | 226.40 | 11.32 | 29.2% | 53.0 | 0.22 / 38.87 | `249.3 / 237.1 / 192.8` |

**B. 搭配 21 餐规则队友（`vs_fastRule`：1 个神经策略 + 2 个 `GpuFastPartners`）**

| 模型 | 队友扰动 | Team Score | Meals | `j_succ` | `j_wait` | 冲突 (资/移) | 分角色 (`S / M / D`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1** (`U200`) | 0% | 413.33 | 20.67 | 78.3% | 43.6 | 0.00 / 1.73 | `420.0 / 400.0 / 420.0` |
| **L1** (`U200`) | **5%** | **367.11** | **18.36** | 65.9% | 43.9 | 0.05 / 9.68 | `376.0 / 356.0 / 369.3` |
| **L1** (`U200`) | **10%** | **326.67** | **16.33** | 55.7% | 46.2 | 0.18 / 14.66 | `336.8 / 323.7 / 319.5` |
| **L2** (`U200`) | 0% | 410.49 | 20.52 | 75.8% | 46.3 | 0.00 / 1.80 | `400.0 / 420.0 / 411.5` |
| **L2** (`U200`) | 5% | 354.84 | 17.74 | 64.1% | 47.5 | 0.03 / 24.40 | `382.7 / 357.3 / 324.5` |
| **L2** (`U200`) | 10% | 308.18 | 15.41 | 53.3% | 52.9 | 0.28 / 34.55 | `341.3 / 317.1 / 266.1` |
| **L3** (`U025`) | **0%** | **417.16** | **20.86** | **77.7%** | **46.9** | 0.00 / 2.50 | **`420.0 / 420.0 / 411.5`** |
| **L3** (`U025`) | 5% | 349.87 | 17.49 | 66.6% | 47.4 | 0.07 / 25.25 | `373.9 / 344.8 / 330.9` |
| **L3** (`U025`) | 10% | 293.78 | 14.69 | 52.8% | 56.9 | 0.20 / 35.09 | `334.9 / 306.1 / 240.3` |

**核心规律**：
1. **L1 → L2 实现跨队友质变**：`all_policy` 从 `72.5` 跃升至 `400.0`（+327.5 分），5%/10% 队友扰动下稳居全场第一（`322.5` / `258.4`）。
2. **L2 vs L3 呈现“鲁棒性 vs 极限上限”权衡**：L3 用纯自博弈低熵收口换取最后 1 餐（`400 → 420`、`vsRule 417.2`），而 L2 因保留了多样化队友池，抗 5%/10% 扰动高出 L3 约 `18 ~ 32` 分。

---

## 二、最终模型是怎么做的

### 1. 网络结构：无历史编码器 + Ego 中心观测重排（[policy.py](mlp_shared/policy.py)）

- **关闭历史编码器（`use_history=False`）**：在 `SharedKitchen` 中，锅具、菜板、装盘台、暂存台与三名厨师的位置/朝向/手持物已完整暴露在当前帧 376 维观测中。我们移除了 `HistoryFeatures`、`EpisodeEncoder`、`PeriodEncoder` 和 `partner_head`，Actor 与 Critic 直接接收当前帧观测，均为 `376 → 256 → 256 → 6/1` 的两层 MLP，总参数量仅 **`326,407`**，既避免历史梯度干扰，又使训练速度提升近 3 倍。
- **Ego 中心观测重排（`ego_reorder`）**：在不改动环境 376 维观测契约的前提下，网络内部根据 `self_id`（`obs[..., 373:376]`）将 3 个厨师的特征块 `obs[..., 1:211]` 重排为 `[self, (self+1)%3, (self+2)%3]`，彻底消除槽位打乱（`shuffle_players=True`）下的对称性歧义，使三角色共享同一策略网络高效收敛。

### 2. 21 餐高效规则队友：`GpuFastPartners`（[fast_partners.py](mlp_shared/fast_partners.py)）

旧版规则队友（`GpuSharedPartners`）使用普通最短路 BFS，每段路到达工位后常多花 1 步原地转向，上限卡在 `19` 餐（`380` 分，普通 13 步 + 联合 18 步 = 31 步/2 餐，`j_wait = 101`）。

新版 `GpuFastPartners` 做了三项核心优化，在 GPU `CUDAGraph` 上达到 **`21.0` 餐 / `420.0` 分**、`j_succ = 100%`、`j_wait = 51.0`、`0/0` 冲突：
1. **朝向感知 0-1 BFS 路由表（`ROUTES_FA`，仅 1.2 MB）**：联合最小化“移动步数 + 到达后朝向对齐转向步数”，把 `D→A_L`（5→4 步）、`S→G_L`（4→3 步）、`G_L→D`（6→5 步）、`K→A_R`（4→3 步）全部压缩至物理下界，稳态节拍精确锁定 **`12`（普通）+ `16`（联合）= `28` 步 / 2 餐**。
2. **GPU 顺序占位防碰撞**：按 `Server → Main → Side` 优先级预订下一拍目标格并阻止对向换位。
3. **右侧装盘与零延误同步卸锅**：Main 在 `G_R` 零延误配合联合卸锅，并在空闲时协助 `A_R` 装盘。

### 3. 纯 0-BC 三阶段训练流程（L1 → L2 → L3）

全程不使用 BC 预训练与 BC 辅助损失，所有阶段均采用 **`num_envs = 2016, rollout_steps = 300`**（与 `horizon = 300` 整局对齐，消除半局奇偶锯齿）、**分角色优势归一化**、**更新步余弦学习率**及 Trackio 实时监控：

1. **阶段一：L1 规则专精基座（[train_l1.py](mlp_shared/train_l1.py)）**
   - 随机初始化 + 100% `GpuFastPartners` 规则队友，`lr = 1e-4`，`entropy_coef = 0.02`，`gamma = 0.995`，`shaping_scale = 0.05`（势能塑形，不改变最优策略），训练 200 轮。
   - 结果：单人搭规则达到 `413.3` 分（`S=420 / M=400 / D=420`），同时自然产出早/中/晚不同水平的 L1 checkpoint（`U=25/30/40/200`）作为 L2 队友池。
2. **阶段二：L2 多样化队友池 + 主自博弈（[train_l2.py](mlp_shared/train_l2.py)）**
   - 从 L1 中 `all_policy` 最高点（`U=40`，`all = 123.6`）初始化，避免从过度特化的末期权重起步。
   - **65% 贪心自博弈（1,308 环境，队友用 `argmax`）+ 35% 早/中/晚 4 权重冻结 L1 队友池（708 环境：`U=25/30/40/200`）**，`lr = 1e-4`，`entropy_coef = 0.008`，`shaping_scale = 0.05`，训练 200 轮。
   - 结果：打通全员自博弈（`all_policy = 400.0`），兼顾对冻结 L1 的适配（`vsL1 = 392.4`），获得最强的抗扰动泛化能力。
3. **阶段三：L3 低熵纯自博弈冲 420（[train_l2.py](mlp_shared/train_l2.py)）**
   - 从 L2 末期权重（`U=200`）初始化，切至 **100% 贪心自博弈（`selfplay_frac = 1.0`）**，关闭势能塑形（`shaping_scale = 0.0`），压低熵正则（`entropy_coef = 0.001`），小学习率 `lr = 5e-5` 精调。
   - 结果：第 **25 轮** 即达成 **`all_policy = 420.0`（21.0 餐）**、`vsRule = 417.2`、`vsL1 = 386.4`，在此甜点早停以防止继续纯自博弈导致遗忘 L1。

### 4. 核心代码与快速复现命令

| 文件 | 作用 |
| :--- | :--- |
| [mlp_shared/env.py](mlp_shared/env.py) | GPU 向量化 + `CUDAGraph` 的 `TorchSharedKitchen`（376 维观测、连续混合订单、势能塑形） |
| [mlp_shared/policy.py](mlp_shared/policy.py) | `V3Policy`（支持 `use_history=False` 无历史模式与 `ego_reorder`） |
| [mlp_shared/fast_partners.py](mlp_shared/fast_partners.py) | 21 餐 / 420 分朝向感知冲突规避规则队友 `GpuFastPartners` |
| [mlp_shared/train_l1.py](mlp_shared/train_l1.py) | L1 训练入口（支持 `--no-history`、`--no-bc`、`--partner-type fast`、Trackio） |
| [mlp_shared/train_l2.py](mlp_shared/train_l2.py) | L2 / L3 训练入口（支持多权重 `--partner-pool`、`--selfplay-frac`、`--no-history`） |
| [mlp_shared/evaluation.py](mlp_shared/evaluation.py) | 统一评估接口（`script` / `policy` / `all_policy`） |
| [mlp_shared/export_replay.py](mlp_shared/export_replay.py) | 导出交互式 HTML / GIF / JSON 回放（如 [fast_bc_21meals.html](mlp_shared/replays/fast_bc_21meals.html)） |

```powershell
# 1. L1（0-BC、无历史、21 餐规则队友）
$env:PYTHONPATH="."
python -m reproduction.nest_rl.mlp_shared.train_l1 `
  --campaign no-hist-nobc-fast-l1-lr1e4-seed1 --no-history --no-bc --partner-type fast `
  --num-envs 2016 --rollout-steps 300 --updates 200 --eval-every 5 `
  --lr 1e-4 --entropy-coef 0.02 --gamma 0.995 --shaping-scale 0.05 --seed 1

# 2. L2（从 L1 U=40 起步，65% 贪心自博弈 + 早/中/晚 4 队友池）
python -m reproduction.nest_rl.mlp_shared.train_l2 `
  --campaign no-hist-l2-from40-sp65-pool4-lr1e4-seed1 --no-history --partner-type fast `
  --init-policy <L1_DIR>/policy_000040.pt `
  --partner-pool <L1_DIR>/policy_000025.pt <L1_DIR>/policy_000030.pt <L1_DIR>/policy_000040.pt <L1_DIR>/policy_000200.pt `
  --selfplay-frac 0.65 --num-envs 2016 --rollout-steps 300 --updates 200 --eval-every 5 `
  --lr 1e-4 --entropy-coef 0.008 --gamma 0.995 --shaping-scale 0.05 --seed 1

# 3. L3（100% 贪心自博弈低熵冲 420，取 U=25 甜点）
python -m reproduction.nest_rl.mlp_shared.train_l2 `
  --campaign no-hist-l3-sp100-420sprint-seed1 --no-history --partner-type fast `
  --init-policy <L2_DIR>/policy_000200.pt --partner-pool <L1_DIR>/policy_000200.pt `
  --selfplay-frac 1.0 --num-envs 2016 --rollout-steps 300 --updates 25 --eval-every 5 `
  --lr 5e-5 --entropy-coef 0.001 --gamma 0.995 --shaping-scale 0.0 --seed 1
```

---

## 三、历史迭代版本归档

以下记录从论文原始复现逐步演进到当前 `mlp_shared` 最终方案的各代历史探索，详细命令与脚本均保留在各自子目录内：

| 迭代阶段 / 目录 | 环境与任务 | 核心做法与主要结论 | 详细文档 |
| :--- | :--- | :--- | :--- |
| **1. 论文原始复现** (`orgin/`) | 作者 Overcooked 2 人沙拉厨房（`fc_small_test.yaml`，73 维观测） | 固定论文 [PDF](NestRL-2602.17737v2.pdf) 与 `adaptive-RL` 子模块（`33bb3222`）；验证公开 PACE 权重（`full` 伙伴下 3 食谱完成率 88%~94%）；打通 L1/L2 训练、多进程/GPU 向量化、空间观测 CNN 与 Trackio 记录。 | [orgin/README.md](orgin/README.md) |
| **2. 标准环境 CNN 对照** (`cnn_standard/`) | 标准 Overcooked 空间网格 | 建立对照目录与规则厨师基线，用于后续空间观测对照。 | [cnn_standard/README.md](cnn_standard/README.md) |
| **3. 标准/复杂环境 MLP 迁移** (`mlp_standard/`, `mlp_complex/`) | 2~3 人传菜与混合订单厨房 | 验证 GPU 向量化厨房与 MLP 策略迁移链路，为后续三人分工环境铺路。 | [mlp_standard/README.md](mlp_standard/README.md) · [mlp_complex/README.md](mlp_complex/README.md) |
| **4. 三人分区厨房 v3** (`mlp_partition_v3/`) | 3 人分区协作厨房 | 引入 `V3Policy`（5×40 两级 MLP 历史编码器，256 隐层）、GPU 规则队友与 BC/课程学习实验；验证了历史编码、学习率调度与回放导出。 | [mlp_partition_v3/README.md](mlp_partition_v3/README.md) |
| **5. 共享厨房早期探索** (`mlp_shared/` 有历史 + BC/DR 版) | `SharedKitchen`（376 维，19 餐旧规则 `GpuSharedPartners`） | 首次将 `V3Policy` 迁移至共享厨房，定位并修复 `CUDAGraph` 奖励别名、跨局角色槽位错配、`t=0` 对称性锁死（引入 `ego_reorder`）与 Critic 梯度冲击（`latent.detach()`）；BC + 域随机化（50% 随机出生 + 5% 噪声 + 35% 自博弈）达到 `19.0` 餐（`380` 分）、全图随机出生 `283.5` 分。受旧 BC 19 餐上限及单边纳什锁制约，由此催生了上文的 **21 餐 `GpuFastPartners` + 无历史 0-BC L1→L2→L3 最终方案**。 | [mlp_shared/](mlp_shared/) |
