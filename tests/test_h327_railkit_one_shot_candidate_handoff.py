"""H327 invented-only adversarial tests: RailKit source batch handoff."""
from pathlib import Path
from unittest.mock import patch
import unittest

from h325.two_step_train_candidates import (
    ROLE, SCENES, produce_two_step_train_candidates,
)

SOURCE = Path(__file__).resolve().parents[1] / "h325" / "railkit.html"


def invented_case_factory():
    seq = iter(range(32))

    def invented_case(browser, url, scene):
        return [
            {
                "png": b"invented-only",
                "rgb_digest": bytes([next(seq)]),
                "task": "invented task",
                "native_label": "invented label",
            }
            for _ in range(2)
        ]
    return invented_case


class OneShotCandidateHandoff(unittest.TestCase):
    def test_one_immutable_batch_only_after_all_scenes(self):
        calls = []

        def sink(batch):
            calls.append(batch)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 32)
            self.assertTrue(all(isinstance(row, tuple) and len(row) == 3
                                for row in batch))
            with self.assertRaises(TypeError):
                batch[0] = ()
            with self.assertRaises(TypeError):
                batch[0][0] = b"changed"
            return True

        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=invented_case_factory()) as source:
            result = produce_two_step_train_candidates(
                object(), SOURCE, role=ROLE, trusted_batch_sink=sink)
        self.assertEqual(source.call_count, len(SCENES))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["candidate_handoff_mode"], "one_shot")
        self.assertEqual(result["train_candidate_messages"], 32)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["model_skill_gain_proven"])

    def test_mid_source_failure_has_zero_external_calls(self):
        calls = []
        count = [0]
        fresh = invented_case_factory()

        def fail_later(browser, url, scene):
            count[0] += 1
            if count[0] == 9:
                raise ValueError("invented source failure")
            return fresh(browser, url, scene)

        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=fail_later):
            with self.assertRaises(ValueError):
                produce_two_step_train_candidates(
                    object(), SOURCE, role=ROLE,
                    trusted_batch_sink=lambda batch: calls.append(batch))
        self.assertEqual(calls, [])

    def test_duplicate_visible_frame_refused_before_consumer(self):
        calls = []

        def duplicate(browser, url, scene):
            return [
                {"png": b"invented", "rgb_digest": b"same",
                 "task": "invented", "native_label": "invented"}
                for _ in range(2)
            ]

        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=duplicate):
            with self.assertRaises(ValueError):
                produce_two_step_train_candidates(
                    object(), SOURCE, role=ROLE,
                    trusted_batch_sink=lambda batch: calls.append(batch))
        self.assertEqual(calls, [])

    def test_missing_or_ambiguous_consumer_refused(self):
        for kwargs in (
            {},
            {"trusted_sink": lambda *args: None,
             "trusted_batch_sink": lambda batch: True},
        ):
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaises(ValueError):
                    produce_two_step_train_candidates(
                        None, SOURCE, role=ROLE, **kwargs)

    def test_nonliteral_acknowledgment_refused(self):
        for ack in (None, False, 1, "true"):
            calls = []
            with self.subTest(ack=ack), patch(
                "h325.two_step_train_candidates.one_source_case",
                side_effect=invented_case_factory()
            ):
                with self.assertRaises(ValueError):
                    produce_two_step_train_candidates(
                        object(), SOURCE, role=ROLE,
                        trusted_batch_sink=lambda batch: (
                            calls.append(batch), ack)[1])
            self.assertEqual(len(calls), 1)

    def test_consumer_exception_never_reports_success(self):
        calls = []

        def fail(batch):
            calls.append(batch)
            raise RuntimeError("invented consumer failure")

        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=invented_case_factory()):
            with self.assertRaises(RuntimeError):
                produce_two_step_train_candidates(
                    object(), SOURCE, role=ROLE, trusted_batch_sink=fail)
        self.assertEqual(len(calls), 1)

    def test_legacy_h325_per_row_behavior_unchanged(self):
        calls = []
        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=invented_case_factory()):
            result = produce_two_step_train_candidates(
                object(), SOURCE, lambda *args: calls.append(args),
                role=ROLE)
        self.assertEqual(len(calls), 32)
        self.assertEqual(result["candidate_handoff_mode"], "legacy_per_row")
        self.assertFalse(result["official_train_admitted"])


if __name__ == "__main__":
    unittest.main()
