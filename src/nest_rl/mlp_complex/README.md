# Soup coordination

同一个 NestRL MLP 在 `soup_coordination` 上训练。默认 64 个 GPU 厨房，两种学习者分工各占 32 个。模型负责备料与开煮时，规则队友负责装盘与送餐，另一半反过来。每局开始独立抽取两个不同的可走格作为出生点，并随机设置朝向。

每局首单和每次正确出餐后的下一单，都从地图定义的五种配方独立均匀抽取。只有当前配方匹配的汤计入出餐。每局至少正确出餐一次计为成功。出餐时间从订单发出计算到正确交付，以环境步为单位。

按[历史 CUDA 环境说明](../mlp_partition_v3/README.md)准备 `.environment/gpu-dev` 后，从项目根目录运行：

```powershell
.\src\nest_rl\orgin\.conda\python.exe .\src\nest_rl\mlp_complex\train_mlp.py --run-name soup-coordination-s1 --seed 1
```

默认训练 960,000 个环境步，每 192,000 步评估并保存策略。Trackio 项目为 `mlp-complex-l1-soup-coordination`。

长跑使用以下参数。64 个厨房每轮各采样 1,024 步，122 轮合计 7,995,392 步。策略权重在跨过每个 200,000 步阈值后的第一个 PPO 更新边界保存，完整训练状态使用 800,000 步阈值。文件名使用实际环境步数。

```powershell
$training = @('--seed', '1', '--learning-rate', '0.0015', '--rollout-steps', '1024',
    '--ppo-mini-batch-size', '1024', '--updates', '122', '--save-every-steps', '200000',
    '--checkpoint-every-steps', '800000', '--eval-every-updates', '5')
.\src\nest_rl\orgin\.conda\python.exe .\src\nest_rl\mlp_complex\train_mlp.py @training --run-name soup-fresh-8m
```

从 `training-state/<实际步数>.pt` 恢复时，沿用原训练参数，传入新的 `--run-name` 和 `--resume <保存点路径>`。完整保存点包含策略、Adam、历史缓冲、厨房状态、当前订单与随机数状态。
