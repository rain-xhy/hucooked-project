# 标准厨房 MLP

在根目录 `hucooked` 的 `cramped_room` 上训练 L1。地图没有中间隔板，使用原始三洋葱汤订单、出生点、交互和碰撞规则。每局 400 步，训练奖励为原环境的出餐分加原环境提供的 shaping 奖励。成功率按每局是否至少一次正分出餐计算。

模型用 60 维浮点观测。字段包含两位厨师的位置、朝向和手持物，设施与台面物件的相对位置，前方地形和物件，四邻格障碍，锅的食材数、煮制进度和局内时间。Actor、Critic 和历史编码器使用 MLP。L0 是 `cnn_standard` 的备料角色，路线和让路表在初始化时生成，训练时用 GPU 张量选动作。

复用 `orgin/.conda`。已有 `hucooked` 的依赖目录时启动脚本会加载它。也可以在该项目环境里安装根目录的 `hucooked`。

```powershell
# 在 mlp_standard 中运行
..\orgin\.conda\python.exe check_parity.py --device cuda
..\orgin\.conda\python.exe train_mlp.py --run-name mlp-cramped-l1-check
```

默认运行一轮 PPO，64 个厨房各采 1024 步，mini-batch 为 1024，4 个 epoch，学习率 `1e-3`，按优化器步数做 2% warmup。Trackio 项目为 `mlp-standard-l1-cramped-room`，日志位于本目录 `runs/trackio`。

## 完整训练与恢复

首次使用续训记录点或延后汇总采样统计时，从主仓库根目录对固定版本的训练器应用补丁。

```powershell
git submodule update --init reproduction/nest_rl/adaptive-RL
git -C reproduction/nest_rl/adaptive-RL apply ../adaptive-rl-training.patch
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
