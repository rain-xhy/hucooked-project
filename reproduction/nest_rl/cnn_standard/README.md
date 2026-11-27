# 标准环境 CNN 路线

此目录包含根目录 `hucooked` 的规则伙伴和评估入口。CNN 训练计划见[复现总览](../README.md)。

在仓库根目录执行以下命令进入标准路线。已有 `.conda/` 时跳过创建步骤。

```powershell
Set-Location reproduction/nest_rl/cnn_standard
conda create --prefix .conda python=3.10 pip -y
conda run --prefix .conda python -m pip install -e ../../../hucooked
```

## 规则厨师

`rule_chefs.py` 提供实现原版 `Agent` 接口的三个角色。

| 角色 | 行为 |
| --- | --- |
| `full` | 负责从取料到出餐的完整流程，按当前订单、锅和手持物选择下一步。 |
| `prep` | 优先拿食材、下锅，或把原料和碗放到共享台面交给队友。空闲时会尝试让出队友的路线。 |
| `serve` | 优先开火、取碗、装汤和出餐。只有自己能把原料送进锅时，也会补上这一步。 |

策略从原版 `OvercookedState` 读位置和物件，输出原版动作。两侧分隔时可以通过共享台面传食材、碗或成品汤。同一时刻优先围绕一口可做有效订单的锅工作。`make_rule_team(mdp)` 根据地图创建规则队伍。少数地图指定了队员分工，其余地图默认使用全能厨师。

## 跑原版任务

在 `cnn_standard/` 中运行。`--roles` 按地图玩家编号顺序传入，一人一个角色。

```powershell
conda run --prefix .conda python evaluate_rule_chefs.py --layout cramped_room --roles prep serve
conda run --prefix .conda python evaluate_rule_chefs.py --layout forced_coordination --roles serve prep
conda run --no-capture-output --prefix .conda python evaluate_rule_chefs.py --all
conda run --prefix .conda python -m unittest discover -v
```

评估使用每张地图的原始开局，默认每局 400 步。`first` 是第一次获得正分出餐的步数，`score` 是原版稀疏奖励，`deliveries` 按玩家统计原版出餐事件。原版 `tutorial_3` 对奖励菜谱返回 `inf`。评估入口用浮点数组累计奖励，保留这个原始分数。
