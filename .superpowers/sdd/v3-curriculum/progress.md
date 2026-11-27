# Sequential role curriculum
User authorized restart from BC, priority left -> right_cook -> right_courier -> joint.
Roles 45/9/9 rotate then21/21/21;63 kitchens horizon400 rollout800, native(20,10,50), potential shaping.
BC rehearsal .1, targetKL .02; first stages lr.001 joint .0003. Budget max20updates perstage, gated advancement two consecutive evals, no blind jump upon failure.
Preserve historical mixed experiment; stop it when replacing. Full latest saved state40, policy50.
Tasks: Trainer extensions; curriculum runner and tests/docs; review; actual GPU launch and verify first update.
Completed: trainer/curriculum implementation, independent review, 43-test full suite passed. Recovery defects fixed: pending promotion and journal truncation.
Actual CUDA run launched final version PID46472, campaign counter-curriculum-seed1; first two updates completed, BC loss .11350, KL early stop active, all metrics finite.
Dashboard localhost7864 hucooked-v3-cooperation HTTP200. Run name counter-curriculum-seed1-ppo-seed1.
Stage1 left45/cook9/courier9. First fixed evaluation atupdate5 pending. Training intentionally left running as user requested; no claim of improved evaluation yet.
