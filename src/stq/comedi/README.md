# CoMeDi

Run from the `hucooked` root:

```bash
CUDA_VISIBLE_DEVICES=1 python -m comedi.train --output comedi/results --steps 16000000 \
  --max-seconds 158400 --population 8 --num-envs 16 --workers 4 \
  --rollout 300 --rounds-per-convention 128 --seed 1 --device cuda:0 \
  --epochs 4 --minibatches 8 --xp-probe-fraction 0.2 \
  --eval-interval 500000 --eval-episodes 12 --save-interval 250000 \
  --demonstrations baseline_common/demonstrations.npz --bc-epochs 64
```

The step budget counts joint environment transitions, including the demonstration
dataset once when it is enabled. Each convention is trained
in rounds: collect self-play, probe prior conventions and select the highest
cross-play return, collect XP against that prior in all six three-role masks,
then collect MixedPlay. The current actor stays fixed during collection; one
PPO update combines positive SP, negatively weighted clipped XP, and positive
MP objectives. MP prefix actions mix old and current actors; only the all-current
suffix is optimized. A fresh actor starts each convention unless
`--warm-start-conventions` is set. When `--demonstrations` is supplied, each
new actor receives a 64-epoch supervised warm start followed by native
SharedKitchen evaluation. The CBR actor receives the same rule initialization;
its population-teacher self-play BC phase remains enabled. The dataset's 9,600
transitions are reserved once from `--steps`, while repeated use of the same
data does not count again.
Half the data uses clean expert trajectories, and half includes independent
0.1-probability chef movement/wait deviations. The saved action is always the
expert correction for the observed state, rather than the injected deviation.
Train and validation episodes are disjoint.

After the population, each convention-aware training round jointly optimizes
an SP PPO buffer and another SP buffer with behavior-cloning labels from all
frozen population actors. Fractions,
round count, XP/MP weights, and the BC coefficient are configurable in
`python -m comedi.train --help`. `--smoke` shrinks the population and budgets but
keeps SP, XP, MP, and CBR in the run.

`config.json` records budgets, SharedKitchen observation dimensions, native
metrics, demonstration identity, and source hashes; `metrics.jsonl` records
separate demonstration and RL step totals, means of native episode metrics,
and phase wall time. `initial.pt`, `latest.pt`,
`best.pt`, and `final.pt` are written atomically. `best.pt` is selected by
SharedKitchen `team_score` validation of the convention-aware actor;
population validation is stored separately as `best_population.pt`. Time-limited runs end with
`status.json` state `stopped` and do not create a completed final artifact.
