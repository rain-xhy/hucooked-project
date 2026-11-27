"""Focused, model-free checks for the final-suite evaluator contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_final_suite.py"
SPEC = importlib.util.spec_from_file_location("stq_evaluate_final_suite", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(evaluate)


class FinalSuiteContractTests(unittest.TestCase):
    def test_delivery_manifest_and_single_gpu_concurrency_defaults(self):
        manifest, _ = evaluate.load_manifest(evaluate.DEFAULT_MANIFEST)
        self.assertEqual(len(manifest["experiments"]), 26)
        args = evaluate.build_parser().parse_args([])
        self.assertEqual(args.gpus, [1])
        self.assertEqual(args.max_parallel, 4)

    def test_manifest_selects_checkpoint_and_only_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected_checkpoint = root / "policy" / "final.pt"
            selected_checkpoint.parent.mkdir()
            selected_checkpoint.write_bytes(b"weights")
            entries = [
                {"experiment_id": "policy_a", "family": "lipo", "result_dir": "policy",
                 "checkpoint": "final.pt", "actual_env_steps": 100, "training_state": "completed"},
                {"experiment_id": "policy_b", "family": "joint_comedi", "result_dir": "b",
                 "checkpoint": "b/final.pt", "actual_env_steps": 200, "training_state": "completed"},
            ]
            path = root / "experiments.json"
            path.write_text(json.dumps({"experiments": entries}), encoding="utf-8")
            manifest, resolved_manifest = evaluate.load_manifest(path, root=root)
            self.assertEqual(resolved_manifest, path.resolve())
            self.assertEqual([x["experiment_id"] for x in evaluate.select_experiments(
                manifest["experiments"], ["policy_b", "policy_a"])], ["policy_b", "policy_a"])
            self.assertEqual(evaluate.resolve_checkpoint_path(entries[0], root=root), selected_checkpoint.resolve())

    def test_manifest_rejects_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "experiments.json"
            path.write_text(json.dumps({"experiments": [
                {"experiment_id": "policy_a", "family": "ppo", "result_dir": "policy",
                 "checkpoint": "best.pt", "actual_env_steps": 100, "training_state": "completed",
                 "backbone": "mlp", "model_size": "small"}
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "best.pt"):
                evaluate.load_manifest(path, root=root)

    def test_payload_steps_and_ppo_size_are_schema_aware(self):
        self.assertEqual(evaluate.extract_checkpoint_steps("lipo", {"total_env_steps": 12}), 12)
        self.assertEqual(evaluate.extract_checkpoint_steps(
            "comedi", {"counters": {"total_train_env_steps": 34}}), 34)
        self.assertEqual(evaluate.ppo_architecture(
            {"backbone": "transformer", "model_size": "million"},
            {"backbone": "transformer", "model_size": "million"}),
            ("transformer", "million"))
        with self.assertRaisesRegex(ValueError, "model_size mismatch"):
            evaluate.ppo_architecture(
                {"backbone": "mlp", "model_size": "large"},
                {"backbone": "mlp", "model_size": "million"})

    def test_native_metrics_keep_exact_fields_and_summary_counts(self):
        raw = {key: 1 for key in evaluate.METRICS}
        raw.update(training_reward=999, training_shaping=50)
        native = evaluate.native_metrics(raw)
        self.assertEqual(tuple(native), evaluate.METRICS)
        self.assertNotIn("training_reward", native)
        summary = evaluate.summarize([native] * 12)
        self.assertEqual(summary["episodes"], 12)
        self.assertEqual(len(summary["episode_metrics"]), 12)
        self.assertEqual(summary["means"]["team_score"], 1.0)
        with self.assertRaisesRegex(RuntimeError, "Missing native metric"):
            evaluate.native_metrics({"team_score": 1})


if __name__ == "__main__":
    unittest.main()
