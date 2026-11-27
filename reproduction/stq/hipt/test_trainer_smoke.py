"""CPU-only two-stage training check using the KitchenBatch-shaped interface."""

import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baseline_common.demonstrations import warm_start
from hipt.behavior_cloning import warm_start_hierarchical
from hipt.model import FlatActorCritic, HIDDEN_SIZE, HiPTPolicy, VECTOR_DIM
from hipt.population import PopulationTrainer
from hipt.train import (
    _configure_args,
    _evaluate_pool_snapshot,
    _select_mid_snapshots,
    parser,
)
from hipt.trainer import HierarchicalTrainer


class _KitchenBatchStub:
    def __init__(self, num_envs, seed, stage="standard", workers=0,
                 train=False, gamma=0.9985):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.ticks = np.zeros(num_envs, dtype=np.int64)
        self.stage = "standard"

    def _observation(self):
        return {
            "spatial": self.rng.normal(size=(self.num_envs, 3, 26, 7, 7)).astype(np.float32),
            "vector": self.rng.normal(size=(self.num_envs, 3, VECTOR_DIM)).astype(np.float32),
            "state": self.rng.normal(size=(self.num_envs, 863)).astype(np.float32),
        }

    def reset(self):
        self.ticks[:] = 0
        return self._observation()

    def step(self, actions):
        self.assert_actions(actions)
        self.ticks += 1
        done = self.ticks >= 65
        infos = []
        for ended in done:
            infos.append({"episode_summary": {
                "team_score": 1.0,
                "meals_completed": 1,
                "normal_completed": 1,
                "joint_completed": 0,
                "joint_successes": 0,
                "joint_attempt_success_rate": 0.0,
                "joint_wait_steps": 0,
                "three_contributor_meals": 0,
                "resource_conflicts": 0,
                "movement_conflicts": 0,
                "ticks": 65,
                "meals_per_100_steps": 100.0 / 65,
                "episode": int(self.ticks[0]),
                "training_reward": 1.0,
                "training_shaping": 0.0,
            }} if ended else {})
        self.ticks[done] = 0
        return self._observation(), np.ones(self.num_envs, dtype=np.float32), done, infos

    def assert_actions(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 3) or np.any(actions < 0) or np.any(actions > 5):
            raise ValueError("Invalid action batch")

    def close(self):
        pass


class TwoStageTrainerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_smoke_archives_each_stage_update(self):
        args = parser().parse_args([
            "--output", "hipt/results/smoke", "--smoke",
            "--demonstrations", "shared_kitchen_demo.npz",
        ])
        total_updates, pool_updates = _configure_args(args)
        batch_size = args.num_envs * args.rollout
        self.assertEqual(total_updates, 2)
        self.assertEqual(pool_updates, 1)
        self.assertEqual(args.checkpoint_interval, batch_size)
        self.assertEqual(args.bc_epochs, 64)
        self.assertEqual(args.epochs, 1)

    def test_default_budgets_count_demonstrations_without_overshooting(self):
        args = parser().parse_args([
            "--demonstrations", "shared_kitchen_demo.npz",
            "--num-envs", "16", "--rollout", "128",
        ])
        total_updates, pool_updates = _configure_args(args)
        batch_size = args.num_envs * args.rollout
        self.assertEqual(args.steps, 8_000_000)
        self.assertEqual(args.pool_steps, 2_000_000)
        self.assertEqual(args.demonstration_env_steps, 9600)
        self.assertEqual(args.output, "hipt/results")
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.max_seconds, 158400)
        self.assertLessEqual(args.demonstration_env_steps + total_updates * batch_size,
                             args.steps)
        self.assertLessEqual(args.demonstration_env_steps + pool_updates * batch_size,
                             args.pool_steps)

    def test_mid_selection_uses_half_native_team_score_and_marks_nonpositive_final(self):
        scores = {
            250_000: np.asarray([-5.0, 9.0]),
            500_000: np.asarray([-8.0, 18.0]),
            750_000: np.asarray([-10.0, 22.0]),
            1_000_000: np.asarray([-10.0, 40.0]),
        }
        metrics = {
            step: {"team_score": values, "normal_completed": values * 0.0}
            for step, values in scores.items()
        }
        selected, result = _select_mid_snapshots(scores, 1_000_000, metrics)
        self.assertEqual(selected, [250_000, 750_000])
        self.assertEqual(result["members"][0]["target_half_team_score"], -5.0)
        self.assertEqual(result["members"][0]["final_team_score"], -10.0)
        self.assertEqual(result["members"][0]["quality"], "qualityfailed")
        self.assertEqual(result["members"][0]["selected_team_score"], -5.0)
        self.assertIn("normal_completed", result["candidate_metric_means"]["750000"])
        self.assertEqual(result["members"][1]["quality"], "measured_positive_final")

    def test_population_bc_initial_eval_reports_each_members_native_meals(self):
        args = type("Args", (), {
            "population": 2,
            "pool_eval_episodes": 1,
            "workers": 0,
            "gamma": 0.9985,
        })()
        models = [FlatActorCritic(), FlatActorCritic()]
        fake_env_module = types.ModuleType("baseline_common.envs")
        fake_env_module.KitchenBatch = _KitchenBatchStub
        with patch.dict(sys.modules, {"baseline_common.envs": fake_env_module}):
            metrics = _evaluate_pool_snapshot(
                models, args, torch.device("cpu"), seed=123,
                episodes_per_member=1,
            )
        self.assertEqual(metrics["team_score"].tolist(), [1.0, 1.0])
        self.assertEqual(metrics["meals_completed"].tolist(), [1.0, 1.0])
        self.assertTrue(all(model.training for model in models))

    def test_population_and_hierarchical_updates_run_on_cpu(self):
        torch.manual_seed(3)
        num_envs, population = 2, 2
        pool = PopulationTrainer(
            _KitchenBatchStub(num_envs, 1), num_envs=num_envs,
            population_size=population, device=torch.device("cpu"), seed=5,
            epochs=1, minibatches=1,
        )
        pool_metrics = pool.collect_and_update(2)
        self.assertEqual(pool_metrics["stage_steps"], 4)
        self.assertEqual(pool_metrics["active_population_members"], 2)
        self.assertTrue(np.isfinite(pool_metrics["jsd"]))
        self.assertTrue(np.all(pool.policy_ids == pool.policy_ids[:, :1]))
        self.assertEqual(set(pool.policy_ids[:, 0].tolist()), {0, 1})

        snapshots = {
            "initial": [copy.deepcopy(model.state_dict()) for model in pool.models],
            "mid": [copy.deepcopy(model.state_dict()) for model in pool.models],
            "final": [copy.deepcopy(model.state_dict()) for model in pool.models],
        }
        pool.close()
        main = HierarchicalTrainer(
            _KitchenBatchStub(num_envs, 2), snapshots, num_envs=num_envs,
            population_size=population, z_dim=4, device=torch.device("cpu"), seed=7,
            epochs=1, minibatches=1, selfplay_fraction=1.0,
            p_min=20, p_max=20, target_env_steps=42, bptt_context=20,
        )
        high_before = main.model.high_actor.weight.detach().clone()
        low_before = main.model.low_actor.weight.detach().clone()
        original_step = main.optimizer.step
        step_calls = [0]

        def count_optimizer_step(*args, **kwargs):
            step_calls[0] += 1
            return original_step(*args, **kwargs)

        main.optimizer.step = count_optimizer_step
        first = main.collect_and_update(20, influence_coef=1.0)
        self.assertEqual(step_calls[0], 1)
        self.assertAlmostEqual(first["low_entropy_coef"], 0.01)
        self.assertAlmostEqual(first["high_entropy_coef"], 0.01)
        self.assertEqual(first["high"]["transitions"], 0)
        self.assertEqual(first["pending_macro_options"], num_envs * 3)
        self.assertFalse(torch.equal(low_before, main.model.low_actor.weight.detach()))
        step_calls[0] = 0
        second = main.collect_and_update(1, influence_coef=1.0)
        self.assertEqual(step_calls[0], 1)
        self.assertAlmostEqual(second["low_entropy_coef"], 0.01 * (1.0 - 40.0 / 42.0))
        self.assertAlmostEqual(second["high_entropy_coef"], 0.01 * (1.0 - 40.0 / 42.0))
        self.assertEqual(second["high"]["transitions"], num_envs * 3)
        self.assertEqual(second["high"]["cross_rollout_transitions"], num_envs * 3)
        self.assertEqual(second["high"]["max_policy_update_lag"], 1)
        self.assertFalse(torch.equal(high_before, main.model.high_actor.weight.detach()))
        main.close()

    def test_population_and_hierarchical_actors_warm_start_before_ppo(self):
        rng = np.random.default_rng(25)
        with tempfile.TemporaryDirectory(prefix="hipt-bc-") as temporary:
            dataset = Path(temporary) / "tiny_demo.npz"
            np.savez_compressed(
                dataset,
                spatial=rng.normal(size=(27, 26, 7, 7)).astype(np.float32),
                vector=rng.normal(size=(27, VECTOR_DIM)).astype(np.float32),
                actions=rng.integers(0, 6, size=27, dtype=np.int64),
                validation=np.repeat([False, False, True], 9),
                episode_lengths=np.asarray([3, 3, 3], dtype=np.int64),
                env_steps=np.asarray(6),
            )
            pool = PopulationTrainer(
                _KitchenBatchStub(1, 4), num_envs=1, population_size=2,
                device=torch.device("cpu"), seed=6, epochs=1, minibatches=1,
                demonstrations_path=dataset, bc_epochs=1,
            )
            self.assertEqual(len(pool.behavior_cloning_metrics), 2)
            self.assertTrue(all("validation_accuracy" in item
                                for item in pool.behavior_cloning_metrics))
            pool.close()

            policy = HiPTPolicy(z_dim=4)
            high_before = policy.high_actor.weight.detach().clone()
            low_before = policy.low_actor.weight.detach().clone()
            metrics = warm_start_hierarchical(
                policy,
                dataset_path=dataset,
                device=torch.device("cpu"),
                seed=7,
                epochs=1,
            )
            self.assertEqual(metrics["demo_env_steps"], 6)
            self.assertEqual(metrics["train_episodes"], 2)
            self.assertEqual(metrics["validation_episodes"], 1)
            self.assertEqual(metrics["sequence_length_min"], 3)
            self.assertEqual(metrics["sequence_length_max"], 3)
            self.assertEqual(metrics["episode_batch_size"], 2)
            self.assertEqual(metrics["chef_stream_batch_sizes"], [6])
            self.assertEqual(metrics["frame_batch_sizes"], [18])
            self.assertEqual(metrics["train_frames"], 18)
            self.assertEqual(metrics["validation_frames"], 9)
            self.assertTrue(torch.equal(high_before, policy.high_actor.weight.detach()))
            self.assertFalse(torch.equal(low_before, policy.low_actor.weight.detach()))
            self.assertEqual(metrics["demo_env_steps"] * 3, metrics["train_frames"])

    def test_population_random_snapshot_is_cloned_before_bc_and_learning(self):
        before_bc = []

        def fake_warm_start(model, forward_logits, *, dataset_path, device, seed,
                            epochs=8, batch_size=512):
            before_bc.append({
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            })
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.25)
            return {"seed": seed, "epochs": epochs}

        with patch("baseline_common.demonstrations.warm_start", side_effect=fake_warm_start):
            pool = PopulationTrainer(
                _KitchenBatchStub(2, 19), num_envs=2, population_size=2,
                device=torch.device("cpu"), seed=23, epochs=1, minibatches=1,
                demonstrations_path=Path("unused_test_dataset.npz"), bc_epochs=1,
            )
        try:
            self.assertEqual(len(before_bc), 2)
            self.assertEqual(len(pool.uncloned_initial_models), 2)
            initial_snapshot = copy.deepcopy(pool.uncloned_initial_models)
            parameter_name = next(name for name, _ in pool.models[0].named_parameters())
            for member_id, (snapshot, random_state) in enumerate(zip(
                    pool.uncloned_initial_models, before_bc)):
                self.assertEqual(snapshot.keys(), random_state.keys())
                for key in snapshot:
                    self.assertTrue(torch.equal(snapshot[key], random_state[key]))
                    self.assertEqual(snapshot[key].device.type, "cpu")
                    self.assertFalse(snapshot[key].requires_grad)
                    self.assertNotEqual(snapshot[key].data_ptr(), random_state[key].data_ptr())
                self.assertFalse(torch.equal(
                    snapshot[parameter_name],
                    pool.models[member_id].state_dict()[parameter_name].detach().cpu(),
                ))

            bc_weights = [model.state_dict()[parameter_name].detach().clone()
                          for model in pool.models]
            pool.collect_and_update(2)
            for member_id, model in enumerate(pool.models):
                self.assertFalse(torch.equal(
                    bc_weights[member_id], model.state_dict()[parameter_name].detach(),
                ))
                for key in initial_snapshot[member_id]:
                    self.assertTrue(torch.equal(
                        pool.uncloned_initial_models[member_id][key],
                        initial_snapshot[member_id][key],
                    ))
        finally:
            pool.close()

    def test_batched_high_context_replay_matches_single_window_replay(self):
        torch.manual_seed(13)
        model = HiPTPolicy(z_dim=4)
        trainer = object.__new__(HierarchicalTrainer)
        trainer.model = model
        trainer.device = torch.device("cpu")
        trainer.bptt_context = 4
        rng = np.random.default_rng(9)

        def history(length):
            return [{
                "spatial": rng.normal(size=(26, 7, 7)).astype(np.float32),
                "vector": rng.normal(size=(VECTOR_DIM,)).astype(np.float32),
                "done": bool(index == 0),
                "h": np.zeros(HIDDEN_SIZE, dtype=np.float32),
                "c": np.zeros(HIDDEN_SIZE, dtype=np.float32),
            } for index in range(length)]

        first, second = history(5), history(3)
        records = [
            {"history": first, "decision_index": 2, "z": 0},
            {"history": first, "decision_index": 4, "z": 2},
            {"history": second, "decision_index": 2, "z": 1},
        ]
        batched_logp, batched_value, batched_entropy = trainer._high_policy_outputs(records)
        reference = []
        for record in records:
            start = max(0, record["decision_index"] - trainer.bptt_context + 1)
            rows = record["history"][start:record["decision_index"] + 1]
            spatial = torch.as_tensor(np.stack([row["spatial"] for row in rows])[:, None])
            vector = torch.as_tensor(np.stack([row["vector"] for row in rows])[:, None])
            done = torch.as_tensor(np.asarray([row["done"] for row in rows], np.float32)[:, None])
            h = torch.as_tensor(rows[0]["h"]).view(1, 1, HIDDEN_SIZE)
            c = torch.as_tensor(rows[0]["c"]).view(1, 1, HIDDEN_SIZE)
            logits, values, _, _, _, _ = model.unroll(spatial, vector, (h, c), done)
            distribution = torch.distributions.Categorical(logits=logits[-1, 0])
            action = torch.tensor(record["z"])
            reference.append((distribution.log_prob(action), values[-1, 0], distribution.entropy()))
        self.assertTrue(torch.allclose(batched_logp, torch.stack([row[0] for row in reference]), atol=1e-6))
        self.assertTrue(torch.allclose(batched_value, torch.stack([row[1] for row in reference]), atol=1e-6))
        self.assertTrue(torch.allclose(batched_entropy, torch.stack([row[2] for row in reference]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
