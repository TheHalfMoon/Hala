"""H156 paired actual native model reward evaluator correctness, offline."""
import copy
import unittest

from h156.audit_neural_native_gain import paired_episodes


class H156RealNativeGainAudit(unittest.TestCase):
    def setUp(self):
        self.cases = []
        self.episodes = []
        for task_i, task in enumerate(("click-test", "enter-text", "choose-list")):
            for i in range(4):
                image_sha = ("%064x" % (task_i * 4 + i + 1))
                self.cases.append({"image_sha256": image_sha, "task": task})
                # Two model-native trial arms on identical task screenshot.
                before = i in (0, 1)
                after = i in (0, 1, 2)
                for role, outcome in (("before", before), ("after", after)):
                    self.episodes.append({
                        "role": role, "task": task, "image_sha": image_sha,
                        "parse_valid": True, "abstained": False,
                        "model_actions_executed": 1,
                        "native_success": outcome,
                        "native_reward_positive": outcome,
                    })

    def test_12_matched_before_after_tasks_and_real_improvement(self):
        result = paired_episodes(self.episodes, self.cases)
        self.assertEqual(result["dev_total"], 12)
        self.assertEqual(result["before_native_success"], 6)
        self.assertEqual(result["after_native_success"], 9)
        self.assertEqual(result["new_wins"], 3)
        self.assertEqual(result["regressions"], 0)
        self.assertEqual(result["paired_exact_sign_p_two_sided"], 0.25)
        self.assertFalse(result["independent_multisite_transfer_established"])
        self.assertEqual(result["per_action_family"]["choose-list"]["tasks"], 4)

    def test_reject_duplicate_missing_or_unseen_task(self):
        for mutate in ("duplicate", "missing", "unseen"):
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(self.episodes)
                if mutate == "duplicate":
                    rows[-1] = dict(rows[0])
                elif mutate == "missing":
                    rows.pop()
                else:
                    rows[-1]["image_sha"] = "f"*64
                with self.assertRaises(ValueError):
                    paired_episodes(rows, self.cases)

    def test_refuse_fake_reward_from_unparsed_or_abstained_action(self):
        for mutate in ("reward_conflict", "unparsed_success", "abstained_success"):
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(self.episodes)
                row = rows[0]
                if mutate == "reward_conflict":
                    row["native_reward_positive"] = not row["native_success"]
                elif mutate == "unparsed_success":
                    row["parse_valid"] = False
                else:
                    row["abstained"] = True
                with self.assertRaises(ValueError):
                    paired_episodes(rows, self.cases)

    def test_zero_difference_does_not_fake_significance(self):
        rows = copy.deepcopy(self.episodes)
        for r in rows:
            r["native_success"] = False
            r["native_reward_positive"] = False
        result = paired_episodes(rows, self.cases)
        self.assertEqual(result["gain_tasks"], 0)
        self.assertEqual(result["paired_exact_sign_p_two_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
