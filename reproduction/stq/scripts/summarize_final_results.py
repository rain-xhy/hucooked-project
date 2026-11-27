"""Build the final CSV and Markdown table from the native evaluation JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def label(item):
    family = item['family']
    mode = 'BC→RL' if item['training_mode'] == 'bc_rl' else '纯RL'
    if family == 'ppo':
        backbone = {'mlp': 'MLP', 'cnn': 'CNN', 'transformer': 'Transformer'}[item['backbone']]
        size = {'small': '原规模', 'large': '扩大版', 'million': '百万参数'}[item['model_size']]
        target = item['requested_env_steps']
        budget = '目标1亿步' if target == 100000000 else f'{target // 10000}万步'
        return f'PPO–{backbone}，{size}，{mode}，{budget}'
    method = {'lipo': 'LIPO', 'comedi': 'CoMeDi', 'hipt': 'HiPT'}[family.removeprefix('joint_')]
    action = '联合动作' if item['action_mode'] == 'joint' else '分开动作'
    return f"{method}，{action}，{mode}，{item['reward_profile']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'experiments.json')
    parser.add_argument('--evaluation', type=Path, default=ROOT/'results/final_evaluation.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    evaluation = json.loads(args.evaluation.read_text(encoding='utf-8'))
    experiments = manifest['experiments']
    rows = {row['experiment_id']: row for row in evaluation['results']}
    assert not evaluation['failed'], evaluation['failed']
    assert len(rows) == len(experiments) == 26
    native = evaluation['protocol']['native_metrics']
    columns = ['experiment_id','label','family','action_mode','training_mode','reward_profile',
               'backbone','model_size','actor_parameters','actual_env_steps','requested_env_steps',
               'training_state','checkpoint','checkpoint_sha256','episodes']
    columns += [f'{key}_{stat}' for key in native for stat in ('mean','std')]
    output = ROOT/'results/final_metrics.csv'
    with output.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for item in experiments:
            result = rows[item['experiment_id']]
            assert result['checkpoint_sha256'] == item['checkpoint_sha256']
            row = {key: item.get(key) for key in columns if key in item}
            row.update(label=label(item),episodes=result['episodes'])
            for key in native:
                row[f'{key}_mean'] = result['means'][key]
                row[f'{key}_std'] = result['stds'][key]
            writer.writerow(row)

    episodes = sum(row['episodes'] for row in rows.values())
    lines = [
        '# 最终主策略统一测评', '',
        f'本表包含全部26组配置，共{episodes}局；所有行均使用清单指定的最终权重，未选择历史最高分权重。', '',
        '环境固定为 `overcooked_env` 的 `de37e0818c2e76311e105ab8689e5162bd84133f`，'
        '使用 SharedKitchen 默认连续订单任务、300步局长、种子100001、贪心动作，每个主策略12局。'
        'LIPO的8个最终成员各12局，以96局等权平均；其他方法每组12局。'
        'HiPT保留原决策方式，包括由固定随机种子生成的20～40步模式持续时间。', '',
        '得分为均值±对局标准差，其余指标为每局均值。原始数据直接来自环境 `info["metrics"]`；'
        '出餐数对应 `meals_completed`，同步餐数对应 `joint_completed`，同步成功率对应 '
        '`joint_attempt_success_rate`，等待步数对应 `joint_wait_steps`。同步成功率是环境定义的'
        '成功次数与尝试步数之比，先按局计算再平均；它不是同步餐占全部出餐的比例。', '',
    ]
    groups = [
        ('论文方法：11组', [item for item in experiments if item['family'] != 'ppo']),
        ('基础PPO：200万步与2000万步，共12组', [item for item in experiments if item['family'] == 'ppo' and item['model_size'] != 'million']),
        ('百万参数PPO：3组提前停止实验', [item for item in experiments if item['model_size'] == 'million']),
    ]
    for heading, group in groups:
        lines += [f'## {heading}', '',
                  '| 实验 | 实际训练步数 | 得分 | 出餐数 | 同步餐数 | 同步成功率 | 等待步数 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for item in group:
            result = rows[item['experiment_id']]
            means, stds = result['means'], result['stds']
            lines.append(f"| {label(item)} | {item['actual_env_steps']:,} | "
                         f"{means['team_score']:.2f} ± {stds['team_score']:.2f} | "
                         f"{means['meals_completed']:.2f} | {means['joint_completed']:.2f} | "
                         f"{means['joint_attempt_success_rate']*100:.2f}% | {means['joint_wait_steps']:.2f} |")
        lines.append('')
    lines += [
        '## 权重与训练步数', '',
        'CoMeDi取最终CBR主策略，HiPT取最终分层策略，PPO取最终actor；LIPO评估整个最终种群。'
        '两组联合动作HiPT训练结束时没有另存 `final.pt`，使用已完成600万主策略步的终端 `latest.pt`。'
        '三组目标1亿步PPO使用用户终止时最后完整更新的 `latest.pt`，其步数已经从checkpoint实际内容核验。', '',
        '原200万步PPO实际完成1,996,800步；BC版包含9,600步训练示范。LIPO因剩余预算不足一次完整种群更新，'
        '在约1600万步边界结束。HiPT总预算由伙伴池与分层主策略共享；其中主策略强化学习为600万步。'
        '训练验证集的2,400步示范不计入9,600步训练示范。', '',
        '原始训练 `status.json` 是停止前的历史记录；提前停止的交付状态与实际步数以 '
        '`experiments.json` 和本次测评checkpoint字段为准。', '',
        '## 提前停止1亿步训练的依据', '',
        '下表来自训练期间固定验证种子120001的12局验证，与上面的种子100001最终测评区分。'
        '“未突破”仅表示验证平均得分未超过此前最高值，不代表其他协作指标或策略参数没有变化。', '',
        '| 基座 | 训练验证最高分 | 首次达到的步数 | 最终checkpoint步数 | 此后未突破的训练步数 | 后续验证次数 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    plateau = json.loads((ROOT/'results/ppo_plateau_summary.json').read_text(encoding='utf-8'))
    for row in plateau['runs']:
        lines.append(f"| {row['backbone']} | {row['first_max_score']:.2f} | {row['first_max_step']:,} | "
                     f"{row['final_step']:,} | {row['steps_after_first_max_to_final_checkpoint']:,} | "
                     f"{row['subsequent_evaluation_count']} |")
    lines += ['', '## 原始数据与复现', '',
              '- [最终逐局测评及LIPO成员分项](results/final_evaluation.json)',
              '- [全部12项原生指标的均值、标准差与配置CSV](results/final_metrics.csv)',
              '- [训练期间PPO验证曲线CSV](results/ppo_training_evaluations.csv)',
              '- [提前停止证据JSON](results/ppo_plateau_summary.json)',
              '- [最终checkpoint清单](experiments.json)', '',
              '这些实验均只有一个独立训练种子。表中的标准差是对局间波动，不能当作多次独立训练的误差。'
              '零分模型的等待步数为0通常表示未进入同步尝试，不能解释为协作效率更高。', '',
              '```bash', 'python scripts/evaluate_final_suite.py --gpus 1 --episodes 12 --seed 100001',
              'python scripts/summarize_final_results.py', '```', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'experiments':len(experiments),'episodes':episodes,'csv':str(output)},ensure_ascii=False))


if __name__ == '__main__':
    main()
