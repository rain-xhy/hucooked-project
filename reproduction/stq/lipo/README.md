# LIPO for SharedKitchen

This package trains a population of complete three-chef joint policies on
SharedKitchen. Each member shares one `2x64` ELU actor across the three chefs,
using an 866-value local vector observation, a role one-hot, and an independent
episode-level categorical `z8`. The centralized SP and partner-indexed XP
critics consume the 863-value state. SharedKitchen's spatial observation
`[E,3,26,7,7]` is validated at the environment boundary but is not used by this
method. The trainer retains PPO/GAE self-play, separate SP and XP critics, the
highest-sampled-return XP selection and sign reversal, and the learned
`q(z | obs, role, action_probs)` MI objective.

Before RL, each actor receives 64 behavior-cloning epochs from the shared
demonstration dataset. The BC callback derives role IDs from the final three
identity values in the native vector observation and samples an independent
random `z8`; spatial input is accepted by the demonstration API but is not
used by the LIPO actor. The dataset's environment-step count is included once
in the total step budget, independent of the number of population members.
Half of the trajectories follow the expert directly. In the other half, each
chef has a 0.1 chance of executing a movement/wait deviation; labels remain
the expert's correction for the actual observed state. This adds recovery and
synchronization timing examples. Training and validation episodes are separate.

Reference implementation: [github.com/51616/marl-lipo](https://github.com/51616/marl-lipo),
specifically `IncompatMAPPOZ`, `IncompatTrainer`, `default.yaml`, and `lipo.sh`.
The run manifest records this mapping, source SHA-256 hashes, model dimensions,
and parameter counts in every checkpoint.

Run from the `hucooked` repository root after its environment package and
`baseline_common` dependencies are available:

```bash
CUDA_VISIBLE_DEVICES=1 python -m lipo.train \
  --output lipo/results --steps 16000000 \
  --demonstrations baseline_common/demonstrations.npz --bc-epochs 64 \
  --num-envs 32 --rollout 300 \
  --population 8 --seed 1 --device cuda:0 --workers 4 \
  --eval-interval 5 --eval-episodes 8 --max-seconds 158400 --xp-candidates 2 \
  --epochs 6 --minibatch-size 256 --lr 0.0005 --checkpoint-interval 250000
```

Training uses the standard SharedKitchen task throughout. `--steps` includes
demonstration transitions once and counts RL vector environment transitions
once per SP or XP rollout, not three times for the simultaneous chef actions.
Validation reports the native SharedKitchen
episode metrics, including `team_score`, `meals_completed`,
`joint_attempt_success_rate`, conflicts, and `meals_per_100_steps`. The best
checkpoint uses mean `team_score`. Interval archives default to every 250,000
requested environment steps and align to a completed population update; use
`--checkpoint-interval` to select another interval. `initial.pt` preserves the random pre-BC population, `bc_initial.pt`
preserves the BC-initialized population, and `final.pt` is saved at stop.
`latest.pt` is saved at each complete population update boundary. XP and MI
coefficients ramp linearly from zero to their configured values over the first
10% of RL steps. A wall-time stop is reported as `stopped` in `status.json`
after its final checkpoint.

For a short end-to-end check, pass `--smoke --device cpu`. Smoke keeps the
requested population (8 by default) and every SP/XP/MI/role mechanism while
shrinking the rollout, PPO epochs, step budget, and evaluation episodes. An
explicit `--population 2` can be used for a minimal local workflow check.

For inference, `load_population(".../best.pt")` returns a wrapper whose
`act(member_id, observations, latent_ids, deterministic=True)` accepts vector
observations shaped `[E,3,866]` and returns simultaneous `[E,3]` actions. Keep
each chef's latent ID fixed until its episode ends; resample only for a reset
environment. The loader accepts only SharedKitchen LIPO v2 checkpoints.
