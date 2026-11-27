import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import torch

from .curriculum_experiment import promotion, run_curriculum, kitchen_counts, evaluation_due, promotion_boundary
from .evaluation import ROLE_NAMES
from .experiment import read
from .policy import V3Policy
from .train import atomic_save


def result(score=10):
    recipes = {r: dict(delivered=1, delivery_rate=.5, mean_delivery_steps=20)
               for r in ("tomato_egg", "potato", "potato_tomato_beef")}
    return dict(mean_score=score, deliveries_per_episode=3, per_recipe=recipes,
                per_role={r: dict(mean_score=score, per_recipe=copy.deepcopy(recipes)) for r in ROLE_NAMES})


class CurriculumTests(unittest.TestCase):
    def test_gate_retention_focus_and_crosswall(self):
        refs = dict.fromkeys(ROLE_NAMES, 10)
        self.assertTrue(promotion(result(), result(0), refs, 0)[0])
        row = result()
        row["per_role"]["left"]["mean_score"] = 8
        self.assertFalse(promotion(row, result(0), refs, 1)[0])
        row = result()
        row["per_role"]["left"]["per_recipe"]["tomato_egg"]["delivered"] = 0
        self.assertFalse(promotion(row, result(0), refs, 2)[0])
        self.assertFalse(promotion(result(), result(), refs, 3)[0])

    def test_pending_promotion_resume_and_journal_trim(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            def fake_eval(model, device, **kwargs):
                return result(0 if kwargs.get("mode") == "wait" else 10)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", side_effect=fake_eval), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()):
                args = dict(device="cpu", horizon=1, rollout_steps=2, epochs=1, minibatch_size=126,
                            num_envs=63, stage_updates=20, eval_env_steps=630)
                run_curriculum(root, policy, stop_after=5, **args)
                old_manifest = read(root / "experiment.json")
                del old_manifest["num_envs"]
                (root / "experiment.json").write_text(json.dumps(old_manifest))
                saved = torch.load(root / "latest.pt", weights_only=False)
                # Reproduce interruption after a successful promotion checkpoint
                # but before next-stage construction.
                saved["curriculum"]["passes"] = 2
                atomic_save(saved, root / "latest.pt")
                with (root / "train.jsonl").open("a") as handle:
                    handle.write('{"global_update":6,"stale":true}\n')
                run_curriculum(root, policy, stop_after=6, **args)
            summary = read(root / "summary.json")
            self.assertEqual(summary["stage"], 1)
            self.assertEqual(summary["global_update"], 6)
            state = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(state["update"], 1)
            self.assertEqual(tuple(state["config"]["role_counts"]), (9, 45, 9))
            self.assertEqual(state["config"]["lr"], .0003)
            self.assertEqual(read(root / "experiment.json")["learning_rates"], [.0003] * 3 + [.0001])
            lines = (root / "train.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertNotIn('"stale"', "".join(lines))

    def test_budget_explains_single_pass_and_explicit_old_learning_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            responses = iter((result(0), result(10), result(10), result(0), result(10)))
            def step(trainer):
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", side_effect=lambda *a, **k: next(responses)), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                run_curriculum(root, policy, device="cpu", horizon=1, rollout_steps=2,
                               stage_updates=10, focus_lr=.001, joint_lr=.0003,
                               num_envs=63, eval_env_steps=630)
            summary = read(root / "summary.json")
            self.assertEqual(summary["status"], "needs_attention")
            self.assertEqual(summary["passes"], 1)
            self.assertIn("(1/2)", summary["reasons"][-1])
            self.assertEqual(read(root / "experiment.json")["learning_rates"], [.001] * 3 + [.0003])

    def test_scaled_lanes_and_dashboard_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            run = Mock()
            def step(trainer):
                self.assertEqual(trainer.roles.bincount().tolist(), [90, 18, 18])
                for role in range(3):
                    counts = trainer.styles[trainer.roles == role].bincount().tolist()
                    self.assertEqual(counts, [30, 30, 30] if role == 0 else [6, 6, 6])
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", return_value=result()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=run), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                run_curriculum(root, policy, device="cpu", horizon=1, rollout_steps=2,
                               num_envs=126, stop_after=1)
            manifest = read(root / "experiment.json")
            self.assertEqual(manifest["num_envs"], 126)
            self.assertEqual(manifest["role_counts"], [[90, 18, 18], [18, 90, 18], [18, 18, 90], [42, 42, 42]])
            self.assertEqual(run.log.call_args.kwargs["step"], 252)
            state = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(state["config"]["num_envs"], 126)

    def test_128_allocation_evaluation_and_history_boundaries(self):
        self.assertEqual(kitchen_counts(128), ((92, 18, 18), (18, 92, 18), (18, 18, 92), (43, 43, 42)))
        self.assertEqual([i for i in range(1, 11) if evaluation_due(i, 128 * 800, 252000, 10)], [3, 5, 8, 10])
        self.assertEqual([i for i in range(1, 11) if promotion_boundary(i, 800, 400)], [5, 10])
        self.assertEqual(kitchen_counts(9), ((3, 3, 3),) * 4)

    def test_resume_two_passes_at_eight_waits_for_tenth_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            evaluations = iter((result(0), result(10), result(10), result(10)))
            def fake_eval(model, device, **kwargs):
                if "mode" in kwargs:
                    return result(0 if kwargs["mode"] == "wait" else 10)
                return next(evaluations)
            def step(trainer):
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", side_effect=fake_eval), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                args = dict(device="cpu", horizon=1, rollout_steps=2, num_envs=128,
                            stage_updates=10, eval_env_steps=630)
                run_curriculum(root, policy, stop_after=8, **args)
                state = read(root / "summary.json")
                self.assertEqual((state["stage"], state["passes"], state["global_update"]), (0, 2, 8))
                run_curriculum(root, policy, stop_after=10, **args)
            state = read(root / "summary.json")
            self.assertEqual((state["stage"], state["global_update"]), (1, 10))
            checkpoint = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(checkpoint["update"], 0)
            self.assertEqual(tuple(checkpoint["config"]["role_counts"]), (18, 92, 18))

    def test_252_budget_and_promotion_at_five(self):
        self.assertEqual(kitchen_counts(252), ((180, 36, 36), (36, 180, 36), (36, 36, 180), (84, 84, 84)))
        self.assertEqual([i for i in range(1, 6) if evaluation_due(i, 252 * 800, 252000, 5)], [2, 3, 4, 5])
        self.assertEqual([i for i in range(1, 6) if promotion_boundary(i, 800, 400)], [5])
        self.assertEqual(252 * 800 * 5, 63 * 800 * 20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            def fake_eval(model, device, **kwargs):
                return result(0 if kwargs.get("mode") == "wait" else 10)
            def step(trainer):
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", side_effect=fake_eval), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                args = dict(device="cpu", horizon=1, rollout_steps=2, num_envs=252,
                            stage_updates=5, eval_env_steps=630)
                run_curriculum(root, policy, stop_after=4, **args)
                state = read(root / "summary.json")
                self.assertEqual((state["stage"], state["passes"], state["global_update"]), (0, 3, 4))
                run_curriculum(root, policy, stop_after=5, **args)
            state = read(root / "summary.json")
            self.assertEqual((state["stage"], state["global_update"]), (1, 5))
            checkpoint = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(checkpoint["update"], 0)
            self.assertEqual(tuple(checkpoint["config"]["role_counts"]), (36, 180, 36))

    def test_252_by_400_sampling_budget_and_boundaries(self):
        self.assertEqual(252 * 400 * 10, 63 * 800 * 20)
        self.assertEqual([i for i in range(1, 11) if evaluation_due(i, 252 * 400, 252000, 10)], [3, 5, 8, 10])
        self.assertEqual([i for i in range(1, 11) if promotion_boundary(i, 400, 400)], [5, 10])
        self.assertFalse(promotion_boundary(8, 400, 400))
        self.assertEqual(kitchen_counts(252)[0], (180, 36, 36))

    def test_1000_step_rollout_boundaries(self):
        self.assertEqual(252 * 1000 * 4, 63 * 800 * 20)
        self.assertEqual([i for i in range(1, 5) if evaluation_due(i, 252 * 1000, 252000, 4)], [1, 2, 3, 4])
        self.assertEqual([i for i in range(1, 5) if promotion_boundary(i, 1000, 400)], [2, 4])
        self.assertFalse(promotion_boundary(5, 1000, 400))
        self.assertTrue(promotion_boundary(10, 800, 400))

    def test_partial_episode_checkpoint_resume_and_four_update_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            evaluations = iter((result(0), result(0), result(10), result(10)))
            def fake_eval(model, device, **kwargs):
                if "mode" in kwargs:
                    return result(0 if kwargs["mode"] == "wait" else 10)
                return next(evaluations)
            def step(trainer):
                self.assertEqual(trainer.tick, (trainer.update * 1000) % 400)
                trainer.tick = (trainer.tick + 1000) % 400
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", side_effect=fake_eval), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.demonstrations", return_value=(None, None)), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.attach_demonstrations"), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                args = dict(device="cpu", horizon=400, rollout_steps=1000, num_envs=252,
                            stage_updates=4, eval_env_steps=252000)
                run_curriculum(root, policy, stop_after=1, **args)
                checkpoint = torch.load(root / "latest.pt", weights_only=False)
                self.assertEqual((checkpoint["update"], checkpoint["tick"]), (1, 200))
                self.assertEqual(checkpoint["curriculum"]["stage"], 0)
                run_curriculum(root, policy, stop_after=3, **args)
                checkpoint = torch.load(root / "latest.pt", weights_only=False)
                self.assertEqual((checkpoint["update"], checkpoint["tick"]), (3, 200))
                self.assertEqual(checkpoint["curriculum"]["stage"], 0)
                run_curriculum(root, policy, stop_after=4, **args)
            checkpoint = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual((checkpoint["update"], checkpoint["tick"]), (0, 0))
            self.assertEqual(checkpoint["curriculum"]["stage"], 1)

    def test_four_update_budget_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "bc.pt"
            atomic_save(dict(model=V3Policy().state_dict()), policy)
            def step(trainer):
                trainer.update += 1
                return dict(mean_score=0)
            with patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.isolated_evaluation", return_value=result(0)), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.demonstrations", return_value=(None, None)), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.attach_demonstrations"), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.open_run", return_value=Mock()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.recorded_steps", return_value=set()), \
                 patch("src.nest_rl.mlp_partition_v3.curriculum_experiment.Trainer.step", step):
                run_curriculum(root, policy, device="cpu", horizon=400, rollout_steps=1000,
                               num_envs=252, stage_updates=4)
            summary = read(root / "summary.json")
            self.assertEqual((summary["status"], summary["stage"], summary["global_update"]), ("needs_attention", 0, 4))
            self.assertTrue(summary["reasons"])


if __name__ == "__main__":
    unittest.main()
