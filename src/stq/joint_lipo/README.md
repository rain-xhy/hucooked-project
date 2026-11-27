# Joint LIPO

Runs the centralized joint-action LIPO variant without changing `lipo/`.
Each actor call consumes one `[E,863]` team-state batch and three per-chef
episode latents, then emits `[E,3,6]` action logits. Population models and Adam
moments stay in a CPU bank; one reusable learner workspace is used for GPU
sampling and updates. The trainer collects every member's SP/XP rollouts before
updating any member, preserving the population snapshot boundary.

Run from the repository root:

```powershell
python -m joint_lipo.train --mode rl
python -m joint_lipo.train --mode bc_rl --demonstrations baseline_common/demonstrations.npz
```

Each command uses `joint_lipo/<mode>/seed_<seed>/results/` unless `--output`
specifies another run directory. Keep each experiment pointed at its own output path.
GPU runs accept `--physical-gpu 1 --memory-fraction 0.035`; the launcher must
expose that physical GPU as logical `cuda:0`.
Defaults retain the eight-member LIPO population, 16M total environment steps,
32 environments, 300-step rollouts, six PPO epochs, 256 minibatch environment
rows, two XP candidates, and a 44-hour limit. BC runs use 64 epochs and 512
action labels per batch; demonstrations count once toward the total step
budget. Pure-RL runs do not open the demonstration dataset.
