# 标准厨房 MLP

在根目录 `overcooked_env` 的 `cramped_room` 上训练 L1。地图没有中间隔板，使用原始三洋葱汤订单、出生点、交互和碰撞规则。每局 400 步，训练奖励为原环境的出餐分加原环境提供的 shaping 奖励。成功率按每局是否至少一次正分出餐计算。

模型用 60 维浮点观测。字段包含两位厨师的位置、朝向和手持物，设施与台面物件的相对位置，前方地形和物件，四邻格障碍，锅的食材数、煮制进度和局内时间。Actor、Critic 和历史编码器使用 MLP。L0 是 `cnn_standard` 的备料角色，路线和让路表在初始化时生成，训练时用 GPU 张量选动作。

复用 `orgin/.conda`，在该项目环境中安装根目录的 `overcooked_env` 及其依赖。

```powershell
# 在 mlp_standard 中运行
..\orgin\.conda\python.exe check_parity.py --device cuda
..\orgin\.conda\python.exe train_mlp.py --run-name mlp-cramped-l1-check
```

默认运行一轮 PPO，64 个厨房各采 1024 步，mini-batch 为 1024，4 个 epoch，学习率 `1e-3`，按优化器步数做 2% warmup。Trackio 项目为 `mlp-standard-l1-cramped-room`，日志位于本目录 `runs/trackio`。

## 完整训练与恢复

首次使用续训记录点或延后汇总采样统计时，从主仓库根目录对固定版本的训练器应用补丁。

```powershell
git submodule update --init src/nest_rl/adaptive-RL
git -C src/nest_rl/adaptive-RL apply ../adaptive-rl-training.patch
```

在本目录执行。每轮采样 40,000 环境步，每 5 轮评估 5 局并保存一次，75 轮共 3,000,000 环境步。

```powershell
$trainingArgs = @('--learning-rate', '0.003', '--updates', '75',
  '--processes', '64', '--rollout-steps', '625', '--ppo-mini-batch-size', '1000',
  '--checkpoint-every-steps', '200000', '--eval-every-updates', '5',
  '--defer-rollout-stats')
..\orgin\.conda\python.exe train_mlp.py @trainingArgs --run-name mlp-3m
```

用原训练参数、一个新的运行名称和训练记录点继续。步数、学习率 warmup、Adam 状态、厨房状态、历史缓冲和随机数状态从该记录点恢复。新运行在 Trackio 中沿用绝对环境步数。

```powershell
..\orgin\.conda\python.exe train_mlp.py @trainingArgs --run-name mlp-3m-resumed `
  --resume .\runs\mlp-3m\training-state\200000.pt
```

`training-state/<步数>.pt` 是续训记录点，只加载可信来源的文件。`ppo/<步数>.pt` 和 `ppo/latest.pt` 保存供评估的策略。

`--defer-rollout-stats` 在采样时直接用 CUDA 张量更新 reset、mask 和历史，在每轮 rollout 结束后批量取回终局统计。续训时可以打开这个选项。`--profile-cuda-stages` 可在 `profile.json` 额外记录 CUDA 流上的分段用时。

## 三人普通台面交接

`--layout three_player_handoff`、`three_player_sync_handoff` 和 `MixedKitchen-v1` 使用历史 CUDA 环境。按[分区厨房的环境准备说明](../mlp_partition_v3/README.md)将其检出到 `.environment/gpu-dev`。在 `three_player_handoff` 中，左边的规则厨师把洋葱放到中央普通台面，L1 学习中间厨师的取料、装锅和开煮，右边的规则厨师取盘、装汤并送餐。两位规则厨师每步在 GPU 上查表选动作。每局 400 步，奖励为出餐分加原生 shaping，至少一次正分出餐即为成功。

每个视角的 80 维观测包含两位队友和中央台面状态。历史编码器将两位队友的动作作为 36 种联合动作输入，Actor 仍输出 6 个动作。

```powershell
..\orgin\.conda\python.exe check_handoff.py --device cuda
..\orgin\.conda\python.exe train_mlp.py @trainingArgs --layout three_player_handoff --run-name handoff-l1-3m
```

在另一个终端从本目录打开 Trackio。续训时保留 `--layout three_player_handoff` 和原训练参数。

```powershell
$env:TRACKIO_DIR = (Resolve-Path .\runs\trackio).Path
$env:TRACKIO_STORAGE_MODE = 'sqlite'
..\orgin\.conda\python.exe -c "import trackio; trackio.show(project='mlp-standard-l1-three-player-handoff', server_port=7863)"
```

## 三人同步交换台微调

`--layout three_player_sync_handoff` 使用原生同步交接规则。左侧厨师持洋葱面对交换台时，80 维观测中的台面物品特征表示该洋葱可交接。中间厨师空手面对交换台，双方在同一步交互即可交接。左右规则厨师在 GPU 上选动作。

`--init-policy` 从普通台面训练的策略文件加载 Actor、Critic 和历史编码器的权重，从零开始记录新地图的环境步数，并创建新的 Adam 优化器。项目名是 `mlp-standard-l2-sync-handoff`。

```powershell
..\orgin\.conda\python.exe check_handoff.py --layout three_player_sync_handoff
..\orgin\.conda\python.exe train_mlp.py --layout three_player_sync_handoff `
  --init-policy .\runs\handoff-l1-3m\ppo\2400000.pt --run-name sync-handoff-l2 `
  --learning-rate 0.0003 --updates 25 --processes 64 --rollout-steps 625 `
  --ppo-mini-batch-size 1000 --checkpoint-every-steps 200000 --eval-every-updates 5 --defer-rollout-stats
```

## MixedKitchen-v1 三角色训练

`--layout MixedKitchen-v1` 使用 compact 厨房。默认 63 个 GPU 厨房各有一个学习者，联合订单厨师、普通订单厨师、协助厨师各占 21 个厨房，另外两位由 GPU 规则队友控制。三个位置共享一个策略。输入包含原生 187 维观测和 3 维角色标记。训练启用原生势函数奖励，系数为 2。

每局至少交付一份普通订单和一份联合订单才计为成功。`sample/mean_delivery_steps` 和 `eval/mean_delivery_steps` 按成功交付的订单统计从生成到上菜的平均环境步数。评估还记录各角色的成功率，每个角色 5 局。

```powershell
..\orgin\.conda\python.exe check_mixed.py
..\orgin\.conda\python.exe train_mlp.py --layout MixedKitchen-v1 --run-name mixed-v1-l1-3roles `
  --updates 25 --save-every-updates 5 --eval-every-updates 5
```
