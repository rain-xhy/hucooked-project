import tempfile
import unittest
from pathlib import Path
from .evaluation import evaluate


class ReplayExportTest(unittest.TestCase):
    def test_nonzero_replay_matches_native_simulation(self):
        import json
        from overcooked_ai_py.cooperative.partition import PartitionConfig, PartitionKitchen
        with tempfile.TemporaryDirectory() as folder:
            report = evaluate(None, seed=20, mode="script", device="cpu", horizon=80,
                              controller_version="serial-v1", recipe_rewards=(20, 10, 50), replay_dir=folder)
            self.assertGreater(report["total_score"], 0)
            data = json.loads((Path(folder) / "left-fifo.ocrep.json").read_text())
            env = PartitionKitchen(PartitionConfig(layout="counter", horizon=80, recipe_rewards=(20, 10, 50)))
            env.reset(seed=20)
            self.assertEqual(data["frames"][0]["state"]["native_state"], {k:v.tolist() for k,v in env.state.items()})
            for segment, frame in zip(data["segments"], data["frames"][1:]):
                _, reward, _, _, info = env.step(segment["actions"])
                self.assertEqual(reward, segment["sparse_reward"])
                self.assertEqual({k:v.tolist() for k,v in info.items()}, segment["native_info"])
                self.assertEqual({k:v.tolist() for k,v in env.state.items()}, frame["state"]["native_state"])
            self.assertGreater(data["frames"][-1]["score"], 0)

    def test_serial_scores_and_replay_match_transitions(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            report = evaluate(None, seed=20, mode="script", device="cpu", horizon=4,
                              controller_version="serial-v1", recipe_rewards=(20, 10, 50), replay_dir=folder)
            self.assertEqual(report["episodes"], 45)
            self.assertIsNone(report["idle_actionable_ingredient_steps"])
            self.assertEqual(report["total_score"], sum(r["total_score"] for r in report["per_recipe"].values()))
            paths = list(Path(folder).glob("*.ocrep.json"))
            self.assertEqual(len(paths), 9)
            data = json.loads(paths[0].read_text())
            self.assertEqual(len(data["frames"]), 5)
            self.assertEqual(len(data["segments"]), 4)
            self.assertEqual(data["header"]["schema_version"], 1)
            self.assertEqual(data["frames"][-1]["score"], sum(s["sparse_reward"] for s in data["segments"]))
            self.assertIn("native_state", data["frames"][0]["state"])
