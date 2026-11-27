# Central-state HiPT

This package implements the three-chef HiPT comparison with one central
observation encoder and one recurrent state per environment. The shared input
is `obs["spatial"][:, 0]` plus the 863-value public team state. Role-conditioned
heads produce separate high-level z distributions and low-level action
distributions for all three chefs.

The three option clocks, durations, per-role influence KL rewards, macro
returns, and high-level GAE streams remain independent. Population policies
also use central inputs and retain the 16-member JSD objective. The 16 trainable
population models and Adam states stay on CPU; a single GPU workspace stages
their outputs and exact joint-loss output gradients one model at a time.
Frozen initial, middle, and final partner policies stay on CPU.

## Run

Run from the repository root. The launcher exposes physical GPU 1 as `cuda:0`.
Use `--mode rl` to skip demonstrations entirely, or `--mode bc_rl` to warm-start
all 16 population actors and the central hierarchical policy.

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
python -m joint_hipt.train `
  --mode bc_rl `
  --output joint_hipt/bc_rl/seed_1/results `
  --demonstrations baseline_common/demonstrations.npz `
  --device cuda:0 --physical-gpu 1 --memory-fraction 0.035 `
  --steps 8000000 --pool-steps 2000000 `
  --num-envs 16 --rollout 300 --population 16 --z-dim 4 `
  --epochs 1 --minibatches 4 --lr 0.00001 --gamma 0.9985 `
  --gae-lambda 0.95 --workers 2 --bc-epochs 64
```

The step budget includes the 9,600 demonstration train transitions once in
`bc_rl`; it is rounded down to complete `num_envs × rollout` updates. Training
uses the legacy native reward profile. Native evaluations run without shaping.
Run artifacts include the pre-BC random pool, immutable pool checkpoints,
per-member midpoint selection, final partner pool, pre/post-BC native
evaluations, `best.pt`, and update-boundary `latest.pt` checkpoints. A claimed
output directory is single-use; this version does not resume a run.
