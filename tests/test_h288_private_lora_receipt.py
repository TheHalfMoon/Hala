"""H288 hostile synthetic-only receipts; no model/media/private GUI output."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from h280.verify_private_native_aggregate import EvidenceRejected
from h287.private_shiftdesk_lora import verified_report
from h288.verify_private_lora_receipt import (
    SOURCE_COMMIT, TRAIN_BLOB, EVAL_BLOB, MODEL_ID, MODEL_REVISION,
    REPORT_NAME, ORIGINAL_SCENES, verify_bytes, verify_private_file,
)


def fake_rows(successes):
    return [{
        "scene": scene, "model_generation": 1,
        "grammar_call": 1, "strict": True,
        "kind": "click", "native_click": True,
        "task_success": bool(success),
    } for scene, success in zip(ORIGINAL_SCENES, successes)]


def fake_emitter():
    return verified_report(
        steps=[{
            "changed": 4, "coordinates_supervised": True,
            "loss_finite": True, "gradient_nonzero": True,
        } for _ in range(8)],
        base=fake_rows((True, False)),
        adapted=fake_rows((True, True)),
        before=("a" * 64, "b" * 64),
        after=("a" * 64, "b" * 64),
        platform={"free_t4": True, "devices": 1},
        train_hashes=frozenset(str(i).zfill(64) for i in range(24)),
    )


def raw(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode()


class H288ClosedPrivateLoRAReceiptTests(unittest.TestCase):
    def test_accept_only_actual_h287_emitter_contract_invented_values(self):
        safe = verify_bytes(raw(fake_emitter()))
        self.assertEqual(safe["paired"]["base"]["task_successes"], 1)
        self.assertEqual(safe["paired"]["adapted"]["task_successes"], 2)
        self.assertEqual(safe["real_adapter_optimizer_steps"], 8)
        self.assertEqual(safe["validated_original_train_candidates"], 24)
        self.assertEqual(safe["paired_delta"], 1)
        self.assertFalse(safe["generalizable_model_gain_established"])
        self.assertFalse(safe["official_training_admitted"])
        for phrase in ('"image"', '"point"', '"raw_text"', '"weights"',
                       '"gold"', '"prompt"'):
            self.assertNotIn(phrase, json.dumps(safe))

    def test_negative_and_zero_delta_allowed_but_never_generalized(self):
        for base, adapted, delta in (
            ((True, True), (False, False), -2),
            ((False, False), (False, False), 0),
            ((False, True), (True, False), 0),
        ):
            with self.subTest(delta=delta):
                doc = fake_emitter()
                doc["paired_pretrained"] = fake_rows(base)
                doc["paired_adapted"] = fake_rows(adapted)
                doc["paired_pretrained_successes"] = sum(base)
                doc["paired_adapted_successes"] = sum(adapted)
                doc["paired_delta"] = delta
                accepted = verify_bytes(raw(doc))
                self.assertEqual(accepted["paired_delta"], delta)
                self.assertFalse(accepted["generalizable_model_gain_established"])

    def test_reject_fake_cuda_optimizer_or_admission(self):
        cases = (
            ("optimizer", lambda d: d.update(research_train_optimizer_steps=7)),
            ("candidate", lambda d: d.update(original_train_candidates_verified=8)),
            ("freegpu", lambda d: d["hardware"].update(free_t4=1)),
            ("devices", lambda d: d["hardware"].update(devices=0)),
            ("updated", lambda d: d.update(changed_lora_tensors_each_step=False)),
            ("coords", lambda d: d.update(coordinate_supervision_each_step=False)),
            ("pairedidentical", lambda d: d.update(same_exact_eval_screens_before_after=False)),
            ("pixelcollision", lambda d: d.update(train_eval_exact_png_disjoint=False)),
            ("fakeood", lambda d: d.update(research_source_exposed_not_blind_ood=False)),
            ("admit", lambda d: d.update(official_train_dev_ood_admitted=True)),
            ("generalize", lambda d: d.update(generalizable_hala_improvement_proven=True)),
            ("release", lambda d: d.update(adapter_weights_published=True)),
            ("raw", lambda d: d.update(raw_predictions_images_points_or_gold_published=True)),
        )
        for label, mutate in cases:
            d = fake_emitter()
            mutate(d)
            with self.subTest(label=label), self.assertRaises(EvidenceRejected):
                verify_bytes(raw(d))

    def test_reject_wrong_original_sources_and_model_version(self):
        for field, val in (
            ("source_commit", "0" * 40),
            ("train_original_blob", "0" * 40),
            ("eval_original_blob", "0" * 40),
            ("model_id", "other"),
            ("model_revision", "0" * 40),
            ("status", "H287_RELEASED"),
        ):
            d = fake_emitter()
            d[field] = val
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                verify_bytes(raw(d))

    def test_reject_fabricated_task_hits_and_grammar(self):
        for key, val in (
            ("model_generation", 0),
            ("grammar_call", 2),
            ("strict", False),
            ("native_click", False),
            ("task_success", 1),
            ("kind", "abstain"),
            ("scene", ORIGINAL_SCENES[1]),
        ):
            d = fake_emitter()
            d["paired_pretrained"][0][key] = val
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_bytes(raw(d))

    def test_reject_wrong_paired_count_and_report_private_raw_fields(self):
        for field, val in (
            ("paired_pretrained_successes", 0),
            ("paired_adapted_successes", 0),
            ("paired_delta", 0),
        ):
            d = fake_emitter()
            d[field] = val
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                verify_bytes(raw(d))
        for level in ("root", "row", "hardware"):
            d = fake_emitter()
            target = d if level == "root" else (
                d["paired_adapted"][0] if level == "row" else d["hardware"])
            target["private_model_coordinates"] = [500, 700]
            with self.subTest(level=level), self.assertRaises(EvidenceRejected):
                verify_bytes(raw(d))

    def test_reject_duplicate_escaped_json_keys_nan_and_oversize(self):
        good = raw(fake_emitter())
        inputs = (
            good.replace(b'"status":', b'"\\u0073tatus":"fake","status":', 1),
            good.replace(b'"paired_delta":1', b'"paired_delta":NaN'),
            b" " * 5000, b"\xff" * 500, b"not-json",
        )
        for bad in inputs:
            with self.subTest(size=len(bad)), self.assertRaises(EvidenceRejected):
                verify_bytes(bad)

    def test_exact_filename_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            correct = root / REPORT_NAME
            correct.write_bytes(raw(fake_emitter()))
            self.assertEqual(verify_private_file(correct)["paired_delta"], 1)
            other = root / "fake.json"
            other.write_bytes(raw(fake_emitter()))
            with self.assertRaises(EvidenceRejected):
                verify_private_file(other)
            try:
                alias = root / REPORT_NAME.replace(".json", "-alias.json")
                alias.symlink_to(correct)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_private_file(alias)


if __name__ == "__main__":
    unittest.main()
