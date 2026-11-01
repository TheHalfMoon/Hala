"""H307 invented-only tests: no real media, labels, GPU or browser."""
import unittest
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from h285.train_only_browser_candidates import ROLE, SOURCE_BLOB
from h307 import one_shot_train_candidate_handoff as unit


@lru_cache(maxsize=32)
def invented_frame(i):
    # Invented in-memory PNG pixels only; no real source screenshot.
    with BytesIO() as stream:
        Image.new("RGBA", (960, 720), (i, 18, 35, 255)).save(
            stream, format="PNG")
        return stream.getvalue()

def valid_proof():
    return {
        "status": "H286_RELATIVE_ALIGNED_FIRSTPARTY_TRAIN_CANDIDATES_ONLY",
        "original_source_blob": SOURCE_BLOB,
        "role": ROLE,
        "actual_browser_supervisions": 24,
        "relative_targets_roundtrip_verified": 24,
        "english_candidates": 12,
        "arabic_candidates": 12,
        "real_model_generations": 0,
        "optimizer_steps": 0,
        "official_train_admitted": False,
        "source_disjoint_blind_ood": False,
        "adapted_gui_gain_proven": False,
        "media_or_labels_saved_to_public_output": False,
    }


def producer(*, stop_at=None, duplicate=False, mutate=None):
    def run(_browser, _source, sink, *, role):
        assert role == ROLE
        for i in range(24):
            if i == stop_at:
                raise RuntimeError("private invented example MUST NOT LEAK")
            sink(invented_frame(0 if duplicate else i),
                 "invented prompt", "invented completion")
        proof = valid_proof()
        if mutate:
            mutate(proof)
        return proof
    return run


class OneShotTrainHandoffTests(unittest.TestCase):
    def call(self, consumer, role=ROLE):
        return unit.handoff_train_candidate_batch_once(
            object(), Path("invented/shiftdesk.html"), consumer, role=role)

    def test_all_24_one_external_call_and_no_official_admission(self):
        calls = []
        def consumer(batch):
            calls.append(batch)
            return True
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer()):
            safe = self.call(consumer)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], tuple)
        self.assertEqual(len(calls[0]), 24)
        self.assertEqual(safe["external_consumer_calls"], 1)
        self.assertFalse(safe["official_train_admitted"])
        self.assertFalse(safe["blind_ood_evaluated"])
        self.assertFalse(safe["external_storage_atomicity_independently_proven"])
        self.assertNotIn("samples", safe)

    def test_mid_source_failure_never_invokes_external_sink(self):
        calls = []
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer(stop_at=13)):
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "SOURCE_BATCH_FAILED") as error:
                self.call(lambda batch: calls.append(batch))
        self.assertEqual(calls, [])
        self.assertNotIn("private invented", str(error.exception))

    def test_duplicate_image_rejected_before_external_sink(self):
        calls = []
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer(duplicate=True)):
            with self.assertRaises(unit.CandidateHandoffRefused):
                self.call(lambda batch: calls.append(batch))
        self.assertEqual(calls, [])

    def test_forged_official_admission_rejected(self):
        calls = []
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer(
                              mutate=lambda d: d.update(
                                  official_train_admitted=True))):
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "UNQUALIFIED"):
                self.call(lambda batch: calls.append(batch))
        self.assertEqual(calls, [])

    def test_bool_as_optimizer_step_is_not_accepted(self):
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer(
                              mutate=lambda d: d.update(
                                  optimizer_steps=False))):
            with self.assertRaises(unit.CandidateHandoffRefused):
                self.call(lambda batch: True)

    def test_wrong_role_refused_without_source_call(self):
        with patch.object(unit, "produce_relative_train_candidates") as src:
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "TRAIN_CANDIDATE_ROLE_ONLY"):
                self.call(lambda batch: True, role="BLIND_OOD")
            src.assert_not_called()

    def test_consumer_failure_is_categorical_and_never_retried(self):
        calls = []
        def fail(_batch):
            calls.append(True)
            raise RuntimeError("invented private raw output")
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer()):
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "EXTERNAL_ROLLBACK_UNVERIFIED") as error:
                self.call(fail)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("invented private raw output", str(error.exception))

    def test_consumer_must_attest_literal_true(self):
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer()):
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "CONSUMER_DID_NOT_ATTEST"):
                self.call(lambda batch: 1)

    def test_incomplete_batch_never_commits(self):
        calls = []
        def incomplete(_browser, _source, sink, *, role):
            for i in range(23):
                sink(invented_frame(i), "invented", "invented")
            return valid_proof()
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=incomplete):
            with self.assertRaisesRegex(unit.CandidateHandoffRefused,
                                        "UNQUALIFIED"):
                self.call(lambda batch: calls.append(batch))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
