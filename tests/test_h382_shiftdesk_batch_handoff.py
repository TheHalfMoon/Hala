"""H382: one-shot ShiftDesk TRAIN-candidate handoff, not data admission.

All 24 objects are invented in memory; no browser media, labels, model weights,
independent rights or blind OOD is used by these source-only tests.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h285.train_only_browser_candidates import (
    ROLE, SCENES, produce_train_only_batch,
)

SOURCE = Path(__file__).resolve().parents[1] / "h285" / "shiftdesk.html"


def invented_case(_browser, _source, _url, scene):
    i = SCENES.index(scene)
    png = b"INVENTED-NOT-A-PNG-" + bytes([i])
    return {
        "scene": scene[0], "lang": scene[1], "png": png,
        "png_sha256": sha256(png).hexdigest(),
        "rgba_sha256": sha256(b"invented-rgba-" + bytes([i])).hexdigest(),
        "user_task": "Invented training instruction " + str(i),
        "completion": "Invented native label " + str(i),
        "source_state_success": True,
    }


class H382OneShotHandoffTests(unittest.TestCase):
    def test_exact_24_immutable_tuples_delivered_once_only_after_validation(self):
        received = []
        def accept(batch):
            received.append(batch)
            return True

        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=invented_case):
            report = produce_train_only_batch(
                object(), SOURCE, role=ROLE, trusted_batch_sink=accept,
            )

        self.assertEqual(len(received), 1)
        batch = received[0]
        self.assertIs(type(batch), tuple)
        self.assertEqual(len(batch), 24)
        self.assertTrue(all(
            type(item) is tuple and len(item) == 3
            and all(type(value) in (bytes, str) for value in item)
            for item in batch
        ))
        self.assertEqual([item[0] for item in batch],
                         [invented_case(None, None, None, s)["png"]
                          for s in SCENES])
        self.assertEqual(report["candidate_handoff_mode"], "one_shot")
        self.assertEqual(report["actual_browser_source_actions_successful"], 24)
        self.assertEqual(report["english_candidates"], 12)
        self.assertEqual(report["arabic_candidates"], 12)
        self.assertIs(report["official_train_manifest_admitted"], False)
        self.assertIs(report["independent_dev_ood_evaluated"], False)
        self.assertIs(report["adapter_model_improvement_proven"], False)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())

    def test_failing_source_scene_never_invokes_complete_consumer(self):
        seen = []
        generated = 0
        def partial_then_fail(browser, source, url, scene):
            nonlocal generated
            generated += 1
            if generated == 17:
                raise ValueError("invented bad source frame")
            return invented_case(browser, source, url, scene)

        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=partial_then_fail):
            with self.assertRaisesRegex(ValueError, "invented bad source"):
                produce_train_only_batch(
                    object(), SOURCE, role=ROLE,
                    trusted_batch_sink=lambda batch: seen.append(batch),
                )
        self.assertEqual(generated, 17)
        self.assertEqual(seen, [])

    def test_duplicate_rgb_or_png_is_rejected_before_handoff(self):
        for kind in ("png_sha256", "rgba_sha256"):
            with self.subTest(kind=kind):
                received = []
                first = None
                def duplicate(browser, source, url, scene):
                    nonlocal first
                    row = invented_case(browser, source, url, scene)
                    if first is None:
                        first = row[kind]
                    elif scene == SCENES[1]:
                        row[kind] = first
                    return row

                with patch("h285.train_only_browser_candidates._candidate_one",
                           side_effect=duplicate):
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        produce_train_only_batch(
                            object(), SOURCE, role=ROLE,
                            trusted_batch_sink=lambda batch: received.append(batch),
                        )
                self.assertEqual(received, [])

    def test_literal_true_ack_only(self):
        for answer in (False, None, 1, "true", [], {}):
            with self.subTest(answer=repr(answer)):
                invoked = []
                def return_bad(batch):
                    invoked.append(batch)
                    return answer
                with patch("h285.train_only_browser_candidates._candidate_one",
                           side_effect=invented_case):
                    with self.assertRaisesRegex(ValueError, "did not acknowledge"):
                        produce_train_only_batch(
                            object(), SOURCE, role=ROLE,
                            trusted_batch_sink=return_bad,
                        )
                self.assertEqual(len(invoked), 1)

    def test_consumer_exception_is_not_retried_or_turned_into_success(self):
        invoked = 0
        def bad_consumer(batch):
            nonlocal invoked
            invoked += 1
            raise RuntimeError("invented consumer transaction abort")

        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=invented_case):
            with self.assertRaisesRegex(RuntimeError, "transaction abort"):
                produce_train_only_batch(
                    object(), SOURCE, role=ROLE,
                    trusted_batch_sink=bad_consumer,
                )
        self.assertEqual(invoked, 1)

    def test_mutually_exclusive_modes_and_existing_legacy_behavior(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "exactly one trusted"):
            produce_train_only_batch(object(), SOURCE, role=ROLE)
        with self.assertRaisesRegex(ValueError, "exactly one trusted"):
            produce_train_only_batch(
                object(), SOURCE, lambda *args: None, role=ROLE,
                trusted_batch_sink=lambda batch: True,
            )
        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=invented_case):
            result = produce_train_only_batch(
                object(), SOURCE, lambda *args: calls.append(args), role=ROLE,
            )
        self.assertEqual(len(calls), 24)
        self.assertEqual(result["candidate_handoff_mode"], "legacy_per_row")

    def test_legacy_consumer_partial_delivery_risk_is_explicit(self):
        seen = []
        def unsafe_old_sink(png, prompt, label):
            seen.append((png, prompt, label))
            if len(seen) == 3:
                raise RuntimeError("invented legacy consumer lost connection")

        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=invented_case):
            with self.assertRaisesRegex(RuntimeError, "lost connection"):
                produce_train_only_batch(
                    object(), SOURCE, unsafe_old_sink, role=ROLE,
                )
        # This is why one-shot transactional consumers must be preferred;
        # the legacy API cannot roll back third-party storage effects.
        self.assertEqual(len(seen), 3)


if __name__ == "__main__":
    unittest.main()
