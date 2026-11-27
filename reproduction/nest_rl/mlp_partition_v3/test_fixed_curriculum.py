"""Fixed-stage progress is independent of scores and survives stage-end interruption."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from .fixed_curriculum_experiment import run_fixed_curriculum
from .experiment import read
from .policy import V3Policy
from .test_curriculum_experiment import result
from .train import atomic_save

MODULE = "reproduction.nest_rl.mlp_partition_v3.fixed_curriculum_experiment"


class FixedCurriculumTests(unittest.TestCase):
    def mocks(self, stack, score, step):
        stack.enter_context(patch(MODULE + ".isolated_evaluation", return_value=result(score)))
        stack.enter_context(patch(MODULE + ".demonstrations", return_value=(None, None)))
        stack.enter_context(patch(MODULE + ".Trainer.attach_demonstrations"))
        stack.enter_context(patch(MODULE + ".open_run", return_value=Mock()))
        stack.enter_context(patch(MODULE + ".recorded_steps", return_value=set()))
        stack.enter_context(patch(MODULE + ".Trainer.step", step))

    def test_high_scores_do_not_promote_early_and_low_scores_finish_all_forty(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            calls = []
            def step(trainer):
                calls.append((trainer.cfg.seed, trainer.update))
                parameter = next(trainer.model.parameters())
                if calls[:-1] and trainer.update == 0:
                    self.assertEqual(trainer.optimizer.state[parameter]["step"].item(), len(calls) - 1)
                trainer.optimizer.state[parameter] = dict(step=torch.tensor(float(len(calls))),
                    exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                trainer.update += 1
                return dict(mean_score=0)
            args = dict(device="cpu", num_envs=9, horizon=1, rollout_steps=2)
            with ExitStack() as stack:
                self.mocks(stack, 1000, step)
                run_fixed_curriculum(root, policy, stop_after=2, **args)
            state = read(root / "summary.json")
            self.assertEqual((state["stage"], state["stage_update"], state["status"]), (0, 2, "running"))
            with ExitStack() as stack:
                self.mocks(stack, 0, step)
                run_fixed_curriculum(root, policy, **args)
                # Re-running a completed campaign cannot add any optimizer updates.
                # Its readable status also recovers if the prior process exited
                # after latest.pt replacement but before those files were written.
                (root / "summary.json").unlink()
                (root / "status.json").unlink()
                run_fixed_curriculum(root, policy, **args)
            state = read(root / "summary.json")
            self.assertEqual((state["stage"], state["global_update"], state["status"]), (3, 40, "complete"))
            self.assertEqual(read(root / "status.json")["status"], "complete")
            self.assertEqual(calls, [(1 + stage * 1000, local) for stage in range(4) for local in range(10)])
            self.assertEqual(len(list((root / "validation").glob("*.json"))), 20)

    def test_checkpoint_after_tenth_update_does_not_repeat_old_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            calls = []
            def step(trainer):
                calls.append((trainer.cfg.seed, trainer.update))
                trainer.update += 1
                return dict(mean_score=0)
            def save_with_boundary_copy(data, path):
                if path.name == "latest.pt" and data["curriculum"]["stage"] == 0 and data["update"] == 10:
                    atomic_save(data, root / "boundary.pt")
                atomic_save(data, path)
            args = dict(device="cpu", num_envs=9, horizon=1, rollout_steps=2)
            with ExitStack() as stack:
                self.mocks(stack, 0, step)
                stack.enter_context(patch(MODULE + ".atomic_save", side_effect=save_with_boundary_copy))
                run_fixed_curriculum(root, policy, stop_after=10, **args)
                state = read(root / "summary.json")
                self.assertEqual((state["stage"], state["stage_update"]), (1, 0))
                # Simulate interruption after the saved stage-0 update-10 state.
                atomic_save(torch.load(root / "boundary.pt", weights_only=False), root / "latest.pt")
                with (root / "train.jsonl").open("a") as journal:
                    journal.write('{"global_update":11,"stale":true}\n')
                run_fixed_curriculum(root, policy, stop_after=11, **args)
            self.assertEqual(calls, [(1, local) for local in range(10)] + [(1001, 0)])
            state = read(root / "summary.json")
            self.assertEqual((state["stage"], state["stage_update"], state["global_update"]), (1, 1, 11))
            self.assertNotIn('"stale"', (root / "train.jsonl").read_text())

    def test_random_start_skips_all_demonstrations_and_resume_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with torch.random.fork_rng():
                torch.manual_seed(17)
                expected = V3Policy().state_dict()
            calls = []
            def step(trainer):
                first = next(trainer.model.parameters())
                if not calls:
                    for key, tensor in trainer.model.state_dict().items():
                        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(first, calls[-1], rtol=0, atol=0)
                self.assertEqual(trainer.cfg.bc_coefficient, 0)
                self.assertIsNone(trainer.demonstrations)
                with torch.no_grad():
                    first.add_(1)
                calls.append(first.detach().clone())
                trainer.update += 1
                return dict(mean_score=0)
            def fake_eval(model, device, **kwargs):
                if model is not None:
                    for key, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
                return result(0)
            args = dict(device="cpu", seed=17, num_envs=9, horizon=1, rollout_steps=2, from_scratch=True)
            with ExitStack() as stack:
                stack.enter_context(patch(MODULE + ".isolated_evaluation", side_effect=fake_eval))
                demo = stack.enter_context(patch(MODULE + ".demonstrations", side_effect=AssertionError("Must not collect demonstrations")))
                attach = stack.enter_context(patch(MODULE + ".Trainer.attach_demonstrations", side_effect=AssertionError("Must not attach demonstrations")))
                stack.enter_context(patch(MODULE + ".open_run", return_value=Mock()))
                stack.enter_context(patch(MODULE + ".recorded_steps", return_value=set()))
                stack.enter_context(patch(MODULE + ".Trainer.step", step))
                # Even a nonexistent BC path must remain untouched in scratch mode.
                run_fixed_curriculum(root, root / "nonexistent-bc.pt", stop_after=1, **args)
                # Evaluation of the updated policy is not expected to match initialization.
                stack.enter_context(patch(MODULE + ".isolated_evaluation", return_value=result(0)))
                run_fixed_curriculum(root, None, stop_after=2, **args)
                demo.assert_not_called()
                attach.assert_not_called()
            self.assertEqual(len(calls), 2)
            manifest = read(root / "experiment.json")
            self.assertIsNone(manifest["initial_policy"])
            self.assertIsNone(manifest["initial_sha256"])
            self.assertEqual(manifest["initialization"], "random")
            self.assertEqual(manifest["bc_coefficient"], 0)
            self.assertTrue((root / "validation-initial.json").exists())
            self.assertFalse((root / "validation-bc.json").exists())
            self.assertFalse((root / "demonstrations.pt").exists())
            initial = torch.load(root / "initial_random.pt", weights_only=False)
            for key, tensor in initial["model"].items():
                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)

    def test_custom_lr_entropy_and_warmup_are_recorded_and_forwarded(self):
        from .train import scheduled_lr
        self.assertAlmostEqual(scheduled_lr(1e-4, 0, 1000, warmup_fraction=0.1, start_factor=0.1), 1e-5)
        self.assertAlmostEqual(scheduled_lr(1e-4, 99, 1000, warmup_fraction=0.1, start_factor=0.1), 1e-4)
        self.assertAlmostEqual(scheduled_lr(1e-4, 500, 1000, warmup_fraction=0.1, start_factor=0.1), 1e-4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            seen = []
            def step(trainer):
                seen.append((trainer.cfg.lr, trainer.cfg.entropy_coef,
                             trainer.cfg.warmup_fraction, trainer.cfg.warmup_start_factor))
                trainer.update += 1
                return dict(mean_score=0)
            args = dict(device="cpu", num_envs=9, horizon=1, rollout_steps=2,
                        focus_lr=1e-4, joint_lr=3e-5, entropy_coef=0.015,
                        warmup_fraction=0.1, warmup_start_factor=0.05)
            with ExitStack() as stack:
                self.mocks(stack, 0, step)
                run_fixed_curriculum(root, policy, stop_after=1, **args)
            manifest = read(root / "experiment.json")
            self.assertEqual(manifest["learning_rates"], [1e-4, 1e-4, 1e-4, 3e-5])
            self.assertEqual(manifest["entropy_coef"], 0.015)
            self.assertEqual(manifest["warmup_fraction"], 0.1)
            self.assertEqual(manifest["warmup_start_factor"], 0.05)
            self.assertEqual(seen, [(1e-4, 0.015, 0.1, 0.05)])


if __name__ == "__main__":
    unittest.main()
