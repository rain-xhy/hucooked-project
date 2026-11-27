"""SharedKitchen native evaluation-summary aggregation checks."""

import unittest

from hipt.evaluate import EPISODE_METRICS, NATIVE_METRICS, summarize_episode_records


class NativeMetricEvaluationTests(unittest.TestCase):
    def test_evaluation_statistics_keep_all_native_metrics_and_episode_records(self):
        first = {key: float(index + 1) for index, key in enumerate(EPISODE_METRICS)}
        first["episode"] = 3
        second = {key: float(index + 3) for index, key in enumerate(EPISODE_METRICS)}
        second["episode"] = 4

        result = summarize_episode_records([first, second])

        self.assertEqual(result["episodes"], 2)
        self.assertEqual(set(result["metric_means"]), set(EPISODE_METRICS))
        self.assertEqual(set(result["metric_stds"]), set(EPISODE_METRICS))
        self.assertTrue(set(NATIVE_METRICS).issubset(result["metric_means"]))
        self.assertEqual(result["metric_means"]["team_score"], 2.0)
        self.assertEqual(result["records"], [first, second])
        self.assertEqual([row["episode"] for row in result["records"]], [3, 4])


if __name__ == "__main__":
    unittest.main()
