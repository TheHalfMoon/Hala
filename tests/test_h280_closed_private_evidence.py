"""H280 synthetic-only, fail-closed evidence parsing; zero private data/GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from h280.verify_private_native_aggregate import (
    EvidenceRejected, MODEL_ID, MODEL_REV, STOCKROOM_SHA, QUEUEBOARD_SHA,
    QUEUEBOARD_BLOB, STOCKROOM_NAME, QUEUEBOARD_NAME, STOCKROOM_STATUS,
    QUEUEBOARD_STATUS, validate_bytes, validate_local_file,
)


def stockroom():
    rows = []
    for i, label in enumerate(("ar_sorted_tape", "ar_sorted_gloves")):
        hit = i == 0
        rows.append({
            "scene": label,
            "real_model_generation_count": 1,
            "real_token_filter_generate_calls": 1,
            "original_screenshot_unchanged": True,
            "format_class": "STRICT_EXACT_VALID",
            "strict_relative_action_valid": True,
            "native_action_kind": "click",
            "native_action_dispatched": True,
            "first_plus_target_hit": hit,
            "firstparty_inventory_task_success": hit,
            "other_rows_unchanged": True,
        })
    return {
        "status": STOCKROOM_STATUS,
        "pinned_original_source_sha": STOCKROOM_SHA,
        "pinned_model_id": MODEL_ID,
        "pinned_model_revision": MODEL_REV,
        "hardware": {"free_t4": True, "devices": 1},
        "real_pretrained_image_generations": 2,
        "rows": rows,
        "strict_relative_actions": 2,
        "native_actions_dispatched": 2,
        "first_plus_target_hits": 1,
        "actual_firstparty_task_success": 1,
        "real_optimizer_updates": 0,
        "no_raw_predictions_points_or_images_saved": True,
        "weights_or_adapters_published": False,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "generalizable_hala_skill_proven": False,
    }


def queueboard():
    rows = []
    for i, label in enumerate(("en_priority_q118", "ar_all_q142")):
        good = i == 0
        rows.append({
            "scene": label,
            "real_generations": 1,
            "grammar_generate_calls": 1,
            "strict_normalized_json": True,
            "native_dispatched": good,
            "firstparty_task_success": good,
            "native_kind": "click" if good else "abstain",
        })
    return {
        "status": QUEUEBOARD_STATUS,
        "source_commit": QUEUEBOARD_SHA,
        "source_html_git_blob": QUEUEBOARD_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": {"free_t4": True, "devices": 1},
        "real_pretrained_image_generations": 2,
        "model_time_grammar_generate_calls": 2,
        "rows": rows,
        "firstparty_task_successes": 1,
        "optimizer_updates": 0,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "model_improvement_proven": False,
        "raw_model_text_points_screenshots_or_weights_published": False,
    }


def encoded(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode()


class H280ClosedPrivateEvidenceTests(unittest.TestCase):
    def test_accepted_stockroom_is_safe_sanitized_only(self):
        result = validate_bytes(encoded(stockroom()), family="stockroom")
        self.assertEqual(result["counts"]["task_successes"], 1)
        self.assertEqual(result["counts"]["target_hits"], 1)
        self.assertEqual(result["real_generations"], 2)
        self.assertFalse(result["independent_blind_ood"])
        self.assertFalse(result["adapter_improvement_proven"])
        text = json.dumps(result)
        for forbidden in ('"point":', '"raw_prediction":', '.png', '"weight":'):
            self.assertNotIn(forbidden, text)

    def test_accepted_queueboard_is_separate_family(self):
        result = validate_bytes(encoded(queueboard()), family="queueboard")
        self.assertEqual(result["counts"]["task_successes"], 1)
        self.assertEqual(result["counts"]["strict"], 2)
        self.assertEqual(result["real_generations"], 2)
        self.assertNotIn("target_hits", result["counts"])
        self.assertFalse(result["independent_blind_ood"])

    def test_cross_family_replay_rejected(self):
        for doc, wrong in ((stockroom(), "queueboard"),
                           (queueboard(), "stockroom")):
            with self.subTest(family=wrong):
                with self.assertRaises(EvidenceRejected):
                    validate_bytes(encoded(doc), family=wrong)

    def test_reject_nested_and_top_level_raw_leaks(self):
        for family, original in (("stockroom", stockroom()),
                                 ("queueboard", queueboard())):
            for level in ("top", "row", "hardware"):
                doc = copy.deepcopy(original)
                location = (doc if level == "top" else
                            doc["rows"][0] if level == "row"
                            else doc["hardware"])
                location["private_model_point"] = [400, 700]
                with self.subTest(family=family, level=level):
                    with self.assertRaises(EvidenceRejected):
                        validate_bytes(encoded(doc), family=family)

    def test_duplicate_json_keys_even_when_escaped(self):
        original = encoded(stockroom())
        raw = original.replace(b'"status":', b'"\\u0073tatus":"sneaked","status":', 1)
        with self.assertRaisesRegex(EvidenceRejected, "duplicate JSON keys"):
            validate_bytes(raw, family="stockroom")

    def test_bad_utf8_size_and_nonstandard_constants(self):
        inputs = (b"not-json", b"\xff" * 200, b" " * 5000,
                  encoded(stockroom()).replace(b'"devices":1', b'"devices":NaN'))
        for bad in inputs:
            with self.subTest(length=len(bad)):
                with self.assertRaises(EvidenceRejected):
                    validate_bytes(bad, family="stockroom")

    def test_stockroom_unsafe_score_or_provenance_rejected(self):
        mutations = (
            ("row_generation", lambda d: d["rows"][0].update(
                real_model_generation_count=0)),
            ("fake_boolean", lambda d: d["rows"][0].update(
                first_plus_target_hit=1)),
            ("grammar", lambda d: d["rows"][1].update(
                real_token_filter_generate_calls=2)),
            ("forced_success", lambda d: d["rows"][1].update(
                firstparty_inventory_task_success=True)),
            ("wrong_scene", lambda d: d["rows"][0].update(scene="other")),
            ("wrong_source", lambda d: d.update(
                pinned_original_source_sha="0" * 40)),
            ("wrong_model", lambda d: d.update(pinned_model_revision="test")),
            ("wrong_total", lambda d: d.update(
                actual_firstparty_task_success=2)),
            ("fake_gpu", lambda d: d["hardware"].update(free_t4=1)),
            ("claim_ood", lambda d: d.update(independent_blind_ood=True)),
            ("claim_gain", lambda d: d.update(
                generalizable_hala_skill_proven=True)),
            ("published", lambda d: d.update(weights_or_adapters_published=True)),
            ("no_pixels", lambda d: d["rows"][0].update(
                original_screenshot_unchanged=False)),
        )
        for label, change in mutations:
            with self.subTest(label=label):
                doc = stockroom()
                change(doc)
                with self.assertRaises(EvidenceRejected):
                    validate_bytes(encoded(doc), family="stockroom")

    def test_queueboard_unsafe_score_or_provenance_rejected(self):
        mutations = (
            ("row_generation", lambda d: d["rows"][0].update(real_generations=0)),
            ("row_grammar", lambda d: d["rows"][0].update(grammar_generate_calls=2)),
            ("wrong_target", lambda d: d["rows"][0].update(scene="Q-999")),
            ("wrong_source", lambda d: d.update(source_commit="0" * 40)),
            ("wrong_blob", lambda d: d.update(source_html_git_blob="0" * 40)),
            ("wrong_model", lambda d: d.update(model_revision="0" * 40)),
            ("wrong_total", lambda d: d.update(firstparty_task_successes=2)),
            ("fake_gpu", lambda d: d["hardware"].update(devices=True)),
            ("faked_success", lambda d: d["rows"][1].update(
                firstparty_task_success=True)),
            ("faked_dispatched", lambda d: d["rows"][1].update(
                native_dispatched=True)),
            ("fake_kind", lambda d: d["rows"][0].update(native_kind=None)),
            ("claim_gain", lambda d: d.update(model_improvement_proven=True)),
            ("claim_ood", lambda d: d.update(independent_blind_ood=True)),
            ("raw_leak", lambda d: d.update(
                raw_model_text_points_screenshots_or_weights_published=True)),
        )
        for label, change in mutations:
            with self.subTest(label=label):
                doc = queueboard()
                change(doc)
                with self.assertRaises(EvidenceRejected):
                    validate_bytes(encoded(doc), family="queueboard")

    def test_local_filename_and_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / STOCKROOM_NAME
            file.write_bytes(encoded(stockroom()))
            got = validate_local_file(file, family="stockroom")
            self.assertEqual(got["counts"]["task_successes"], 1)
            wrong = root / "wrong-name.json"
            wrong.write_bytes(encoded(stockroom()))
            with self.assertRaises(EvidenceRejected):
                validate_local_file(wrong, family="stockroom")
            with self.assertRaises(EvidenceRejected):
                validate_local_file(file, family="queueboard")
            try:
                link = root / "alias.json"
                link.symlink_to(file)
            except (NotImplementedError, OSError):
                return
            with self.assertRaises(EvidenceRejected):
                validate_local_file(link, family="stockroom")


if __name__ == "__main__":
    unittest.main()
