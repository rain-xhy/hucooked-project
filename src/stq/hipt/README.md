# HiPT on SharedKitchen

HiPT trains a diverse population of 16 partner policies, then trains one shared
hierarchical policy for all three chefs. Each partner policy controls all roles
in its episode and is optimized with PPO plus a Jensen–Shannon diversity
objective. The partner pool keeps three skill levels: randomly initialized
low-skill policies, per-member mid-training snapshots near half the final
standard `team_score`, and fully trained policies. Population BC warm-starts
the active learners but does not replace the frozen random initial policies.
These skill levels follow [HiPT Section 4.1](https://arxiv.org/html/2305.16708).
During hierarchical training, half of episodes are three-chef self-play;
otherwise two chefs use the current HiPT policy and one uses a fixed pool
partner. The partner role rotates between episodes.

All actors use the native SharedKitchen observation: a 26-channel 7×7 spatial
tensor and an 866-value public vector. Their spatial encoder keeps three
convolutions with 5, 3, and 3 kernels, with 16 feature channels. The flattened
spatial features and vector feed two 128-wide layers. HiPT adds a 128-wide LSTM
and 64-wide high- and low-level heads. The option has four discrete values and
holds for a uniformly sampled 20–40 primitive steps.

The high-level option reward averages task reward plus the scheduled HiPT
influence KL. Low and high PPO objectives update the shared parameters together.
The 300-step rollout BPTT and high-level history replay retain recurrent
gradients, while unfinished options keep their timer and accumulated reward
across rollout updates. The population uses 16 partner distributions for JSD;
its middle snapshot is selected per member from the saved 250,000-step
snapshots, nearest half the final standard `team_score`. Final and periodic
evaluation report all native SharedKitchen episode metrics, and `best.pt` is
selected by mean unshaped standard `team_score`.

Each active population learner and the hierarchical actor are warm-started
from the same demonstration dataset before their PPO optimizers are created.
The pre-BC random pool snapshot remains available as the low-skill partner.
Population behavior cloning uses shuffled 512-observation batches.
Hierarchical behavior cloning
uses complete episodes: each two-episode minibatch carries the LSTM state
through six chef streams, or 1,800 frames for 300-step episodes. Teacher actions
supervise every low-level z head; the high-level z head is left to PPO. Both
stages default to 64 BC epochs. The dataset must include `episode_lengths` and
separate train/validation episodes; missing sequence metadata is an error.
Half the trajectories are clean and half include 0.1-probability movement/wait
deviations per chef. All labels are expert corrections for the actual state;
sequence BC therefore includes recovery and altered synchronization timing.

The dataset's 9,600 environment steps count once in the 8,000,000-step run
budget, independent of how many actor weights are fit. Population training
targets 2,000,000 steps including that demonstration budget; the remaining
steps are hierarchical PPO. Training and evaluation use only the `standard`
task.

## Run

From the SharedKitchen repository root, on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python -m hipt.train \
  --output hipt/results \
  --demonstrations baseline_common/demonstrations.npz \
  --device cuda:0 --seed 1 \
  --steps 8000000 --pool-steps 2000000 \
  --num-envs 16 --rollout 300 --workers 4 --epochs 1 \
  --population 16 --z-dim 4 --checkpoint-interval 250000 \
  --bc-epochs 64 --lr 0.00001 --max-seconds 158400
```

The requested step total includes demonstration transitions and both RL stages.
The default output is `hipt/results`. RL work is rounded down to complete
rollout batches. The 44-hour cap includes evaluation. Resume from a checkpoint
inside that run directory with `--resume`.

`config.json` records the demonstration hash and budget, architecture, native
metric names, training protocol, and source hashes. `metrics.jsonl` records
updates. Evaluation JSON files retain episode summaries and mean/standard
deviation for every native metric. Checkpoints contain model and optimizer
states plus the random-initial/mid/final partner pool; `partner_pool.pt` is the hashed
artifact referenced by the manifest. `pool_initial_policy.pt` stores the random
pre-BC population for low-skill partners, while `pool_initial.pt` stores the
complete post-BC learner state for resume. `pool_bc_initial_eval.json` evaluates
the active BC-initialized learners. `initial_uncloned.pt` and `bc_initial.pt`
save the hierarchical policy immediately before and after BC. Their matched
`pre_bc_eval.json` and `bc_initial_eval.json` files include native metrics for
the same seeded episodes.
