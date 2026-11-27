import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from .bc_experiment import compact_metrics, repair_validation_log, RECIPES
from .experiment import save
from .evaluation import ROLE_NAMES


class CompactMetricsTests(unittest.TestCase):
    def test_resume_repairs_persisted_but_unpublished_validation_once(self):
        result = dict(mean_score=120., deliveries_per_episode=5.,
                      per_role={role: dict(mean_score=120.) for role in ROLE_NAMES},
                      per_recipe={name: dict(delivery_rate=.4, mean_delivery_steps=42.) for name in RECIPES})
        wait = dict(mean_score=100., per_role={role: dict(mean_score=100.) for role in ROLE_NAMES})
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save(folder / "validation/020.json", result)
            published, run = set(), Mock()
            for _ in range(2):
                repair_validation_log(folder, 20, 50_400, published, run, wait, dict(mean_score=300.))
            run.log.assert_called_once()
            self.assertEqual(run.log.call_args.kwargs["step"], 1_008_000)
            self.assertEqual(run.log.call_args.args[0]["eval/model_contribution"], 20.)
            self.assertEqual(published, {1_008_000})

    def test_dashboard_reports_actual_contribution_and_keeps_diagnostics_out(self):
        result = dict(mean_score=120., deliveries_per_episode=5.,
                      per_role={role: dict(mean_score=score) for role, score in zip(ROLE_NAMES, (30, 150, 180))},
                      per_recipe={name: dict(delivery_rate=.4, mean_delivery_steps=42.) for name in RECIPES})
        wait = dict(mean_score=100., per_role={role: dict(mean_score=score)
                                             for role, score in zip(ROLE_NAMES, (0, 150, 150))})
        values = compact_metrics(dict(mean_score=90, mean_shaping_reward=2, entropy=1, grad_norm=2),
                                 result, wait, dict(mean_score=300))
        self.assertEqual(values["eval/model_contribution"], 20)
        self.assertEqual(values["contribution/left"], 30)
        self.assertEqual(values["contribution/right_cook"], 0)
        self.assertEqual(values["train/score_per_episode"], 90)
        self.assertEqual(values["train/process_reward_per_episode"], 2)
        self.assertEqual(len(values), 16)
        self.assertFalse(any("loss" in key or "entropy" in key or "grad" in key for key in values))

    def test_undefined_delivery_latency_is_not_shown_as_zero(self):
        result = dict(mean_score=0, deliveries_per_episode=0,
                      per_recipe={name: dict(delivery_rate=None, mean_delivery_steps=None) for name in RECIPES})
        self.assertEqual(compact_metrics(result=result), {"eval/score_per_episode": 0,
                                                         "eval/deliveries_per_episode": 0})


if __name__ == "__main__":
    unittest.main()
