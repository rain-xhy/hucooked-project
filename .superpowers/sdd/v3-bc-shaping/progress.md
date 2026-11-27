# V3 BC + shaped PPO implementation ledger
User authorized BC pretraining, then training, NestRL-inspired process reward, compact Trackio on 7864.
Existing dirty wjp workspace contains entire uncommitted V3 implementation; continue in place to preserve current code and results.
Tasks: BC script/dataset and validation; shaping and exact restart; compact campaign/dashboard; integration tests; run BC then start PPO.
Defaults: seed 1, 63 lanes, 400 horizon, 800 rollout, lr .001, 80 PPO updates, parallel-v2, native recipe rewards (20,10,50).
Metrics: isolated compact project, keep historical project and full local JSON diagnostics. Evaluation native score only and paired waiting contribution by role.
Status: three implementation agents working; parent integration/review/launch.
Completed: BC collection/training and compact dashboard; potential-shaped PPO started on CUDA.
Validation: full V3 suite 36 passed, final changed BC/campaign tests 7 passed; real CUDA fit smoke and exact restart tests passed.
Review: fixed overlapping BC RNG ranges and missing Trackio checkpoint publication repair. No default-run blockers.
Run: runs/counter-bc-shaped-seed1; PID 37728; dashboard PID 48632; localhost7864 project hucooked-v3-cooperation.
BC best epoch8, heldout action accuracy .962611, balanced .946387, NLL .141042.
Paired 45-episode initial evaluation: BC160.444 vswait158.889 vs script320. PPO2/80 observed finite metrics, train native mean182.063.
PPO is intentionally left running as requested (80 updates), not claimed complete.
