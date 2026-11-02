"""H359 actual ephemeral source-only corpus → H199 action message audit.

No adapted Qwen, rights grant, training output, PRIVATE OOD or extra GPU.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, parse_actions,
    training_messages_for_synthetic_contract,
)
from h357.source_candidate import capture_source_candidate
from h358.source_candidate import capture_candidate
from h359.native_candidate_bridge import (
    PREFIX, ROLE357, ROLE358, SOURCES, SCHEMA, audit_review_bundle,
)
from training.artifacts import validate_action_completion


class H359NativeOnlyBridgeTests(unittest.TestCase):
    def test_source_identity_and_two_incompatible_action_contracts(self):
        self.assertEqual(len(SOURCES), 2)
        self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)
        self.assertEqual({x["records"] for x in SOURCES.values()}, {64, 128})
        self.assertEqual({x["role"] for x in SOURCES.values()},
                         {ROLE357, ROLE358})
        for source in SOURCES.values():
            self.assertEqual(
                sha256(source["source"].read_bytes()).hexdigest(),
                source["source_sha"])
        example = '{"actions":[{"kind":"click","point":[245,250]}]}'
        self.assertEqual(parse_actions(example, training_target=True)
                         ["actions"][0]["kind"], "click")
        with self.assertRaises(ValueError):
            validate_action_completion(example)
        with self.assertRaises(ValueError):
            parse_actions('{"type":"click","point":[245,250]}',
                          training_target=True)
        q, a = training_messages_for_synthetic_contract("Choose a tile.", example)
        self.assertEqual(q[0]["content"][0], {"type":"image"})
        self.assertEqual(a[:-1], q)
        self.assertIn('"actions"', a[-1]["content"][0]["text"])

    def test_empty_wrong_and_forged_bundle_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in ("fake-family", 1, None):
                with self.subTest(family=family), self.assertRaises(ValueError):
                    audit_review_bundle(root, family)
            with self.assertRaises(ValueError):
                audit_review_bundle(root, "controlgarden-single-app")
            src = Path(tmp) / "fake"
            src.mkdir()
            (src / "candidate-manifest.json").write_text(
                json.dumps({"schema":SOURCES["controlgarden-single-app"]["schema"],
                            "role":ROLE357,"official_train_admitted":True}),
                encoding="utf-8")
            (src / "candidate-rows.jsonl").write_text("",encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_review_bundle(src, "controlgarden-single-app")

    @unittest.skipUnless(os.environ.get("HALA_H359_BROWSER_QA") == "1"
                         and importlib.util.find_spec("playwright") is not None
                         and importlib.util.find_spec("PIL") is not None,
                         "real browser+Pillow opt-in; pinned in dedicated H359 CI")
    def test_all_192_real_source_actions_roundtrip_and_tampering_fail_closed(self):
        with TemporaryDirectory() as tmp:
            roots = {}
            browser = "msedge" if os.name == "nt" else "chromium"
            first = Path(tmp) / "controlgarden"
            second = Path(tmp) / "noterelay"
            capture_source_candidate(first, browser_kind=browser)
            capture_candidate(second, browser_kind=browser)
            roots["controlgarden-single-app"] = first
            roots["noterelay-original-multistep"] = second
            audits = [audit_review_bundle(root, family)
                      for family, root in roots.items()]
            self.assertEqual([a["records"] for a in audits], [64,128])
            self.assertEqual([a["h199_messages_checked"] for a in audits], [64,128])
            self.assertEqual(sum(a["distinct_images"] for a in audits), 192)
            self.assertEqual(sum(a["en"] for a in audits), 96)
            self.assertEqual(sum(a["ar"] for a in audits), 96)
            self.assertEqual(audits[0]["action_counts"], {"click":64})
            self.assertEqual(audits[1]["action_counts"],
                             {"click":96, "type_text":32})
            for report in audits:
                self.assertEqual(report["schema"], SCHEMA)
                for key in ("legacy_schema_compatible", "official_train_admitted",
                            "independent_rights_verified", "can_train",
                            "blind_ood", "can_release"):
                    self.assertIs(report[key], False)
            # Falsifying a rights field must NOT grant admission.
            for root in roots.values():
                manifest = root / "candidate-manifest.json"
                old = manifest.read_bytes()
                try:
                    forged = json.loads(old)
                    forged["official_train_admitted"] = True
                    manifest.write_text(json.dumps(forged), encoding="utf-8")
                    family = "controlgarden-single-app" if root == first else "noterelay-original-multistep"
                    with self.assertRaises(ValueError):
                        audit_review_bundle(root, family)
                finally:
                    manifest.write_bytes(old)
            # Altering model-supervised native completion is blocked.
            path = first / "candidate-rows.jsonl"
            old = path.read_bytes()
            try:
                rows = [json.loads(line) for line in old.splitlines()]
                rows[0]["completion"] = '{"type":"click","point":[100,120]}'
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
                    encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_review_bundle(first, "controlgarden-single-app")
            finally:
                path.write_bytes(old)
            # Media content changes are rejected by exact bytes.
            file = first / json.loads(old.splitlines()[0])["image_path"]
            original = file.read_bytes()
            try:
                file.write_bytes(original + b"alteration")
                with self.assertRaises(ValueError):
                    audit_review_bundle(first, "controlgarden-single-app")
            finally:
                file.write_bytes(original)
            # After restoring, audits are exact and read-only.
            self.assertEqual(audit_review_bundle(first, "controlgarden-single-app"),
                             audits[0])


if __name__ == "__main__":
    unittest.main()
