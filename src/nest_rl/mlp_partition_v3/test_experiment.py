import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from . import experiment as exp


class CampaignTests(unittest.TestCase):
    def test_fixed_training_budget(self):
        from .train import Settings
        settings = Settings(.001, 1, 80)
        self.assertEqual((settings.horizon, settings.rollout_steps), (exp.HORIZON, exp.ROLLOUT))
        self.assertEqual(exp.SAMPLES, 50_400)
        self.assertEqual(settings.num_envs * settings.rollout_steps, exp.SAMPLES)
        self.assertEqual(exp.SAMPLES * settings.updates, 4_032_000)

    def test_resume_repairs_missing_validation_and_telemetry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "screen-lr0.001-seed1"
            folder.mkdir()
            checkpoint = folder / "checkpoint_000020.pt"
            checkpoint.touch()
            losses = [dict(update=i, loss=i/100) for i in range(1, 21)]
            (folder / "train.jsonl").write_text(
                "\n".join(json.dumps(x) for x in losses) + '\n{"partial":', encoding="utf-8")
            result = dict(macro_delivery_rate=.4, cross_delivery_rate=.3, deliveries_per_episode=3,
                          per_recipe={}, per_role={}, per_style={}, per_episode=[])
            for update in (5, 10, 15):
                exp.save(folder / "validation" / f"{update:03}.json", result)
            published = {i * exp.SAMPLES for i in range(1, 20) if i != 15}
            run, model = Mock(), Mock()
            with (patch.object(exp, "recorded_steps", return_value=published),
                  patch.object(exp, "open_run", return_value=run),
                  patch.object(exp, "evaluate", return_value=result),
                  patch.object(exp, "load_policy", return_value=model),
                  patch("src.nest_rl.mlp_partition_v3.train.train") as train):
                exp.fit(root, "screen", .001, 1, 20)
            self.assertEqual(train.call_args.kwargs["resume"], checkpoint)
            self.assertEqual([c.kwargs["step"] for c in run.log.call_args_list],
                             [exp.SAMPLES*15, exp.SAMPLES*20])
            self.assertEqual(run.log.call_args_list[-1].args[0]["loss"], .2)
            self.assertEqual(exp.read(folder / "validation/020.json"), result)
