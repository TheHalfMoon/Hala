import math
import unittest

from experiments.cost.calculator import CostInputs, estimate, scenario_table


class CostTests(unittest.TestCase):
    def test_work_and_idle_are_accounted_separately(self):
        inputs = CostInputs(users=10, tasks_per_user=100, model_calls_per_task=2,
                            gpu_seconds_per_call=3, retry_rate=0.1,
                            success_rate=0.8, gpu_hourly_usd=1,
                            warm_workers=1, month_hours=720,
                            storage_gb=10, storage_usd_per_gb=0.1,
                            network_gb_per_task=0.001, network_usd_per_gb=0.2,
                            failure_cost_usd=0.01)
        result = estimate(inputs)
        self.assertAlmostEqual(result['active_gpu_hours'], 6600 / 3600)
        self.assertAlmostEqual(result['total_usd'], 723.2)
        self.assertAlmostEqual(result['usd_per_verified_task'], 723.2 / 800)
        self.assertEqual(result['evidence'], 'scenario_assumptions')

    def test_zero_success_has_no_finite_cost_per_success(self):
        result = estimate(CostInputs(success_rate=0))
        self.assertIsNone(result['usd_per_verified_task'])

    def test_invalid_inputs_are_rejected(self):
        for kwargs in ({'users': True}, {'users': -1}, {'success_rate': 1.1},
                       {'gpu_seconds_per_call': math.nan}, {'month_hours': 0},
                       {'gpu_hourly_usd': math.inf}, {'concurrency': 0},
                       {'model_calls_per_task': 1e308}, {'users': 10**100},
                       {'success_rate': 5e-324}, {'month_hours': 5e-324}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate(CostInputs(**kwargs))

    def test_scenarios_cover_requested_counts_and_prices(self):
        rows = scenario_table(CostInputs())
        self.assertEqual(len(rows), 25)
        self.assertEqual({row['users'] for row in rows}, {10, 50, 100, 500, 1000})
        self.assertEqual({row['price_usd'] for row in rows}, {5, 8, 10, 15, 20})


if __name__ == '__main__':
    unittest.main()
