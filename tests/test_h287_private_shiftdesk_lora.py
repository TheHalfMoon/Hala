"""H287 adversarial SOURCE-ONLY tests; no real model, no private logs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h287.private_shiftdesk_lora import (
    BRANCH, PIN, SHIFT_BLOB, QUEUE_BLOB, MODEL, MODEL_REV,
    N_STEPS, STEP_INDICES, STATUS, FLAG, QUEUE_SCENES, REPORT,
    queue_task, model_prompt, verified_report, verified_free_t4,
    validate_training_evidence,
)
from h287.build_private_payload import build, KERNEL_ID, KERNEL_TITLE
from h282.private_clinicgrid_pretrained_probe import RELATIVE_SUFFIX
from h285.train_only_browser_candidates import ROLE


def paired_rows(good=(True, False)):
    return [{
        "scene": scene[0], "model_generation": 1, "grammar_call": 1,
        "strict": True, "kind": "click",
        "native_click": True, "task_success": hit,
    } for scene, hit in zip(QUEUE_SCENES, good)]


def steps():
    return [{
        "changed": 32, "coordinates_supervised": True,
        "loss_finite": True, "gradient_nonzero": True,
    } for _ in range(N_STEPS)]


def verified_kwargs():
    return dict(
        steps=steps(),
        base=paired_rows((True, False)),
        adapted=paired_rows((True, True)),
        before=("a" * 64, "b" * 64),
        after=("a" * 64, "b" * 64),
        platform={"free_t4": True, "devices": 1},
        train_hashes=frozenset(str(i).zfill(64) for i in range(24)),
    )


class H287OriginalCrossfamilySafetyTests(unittest.TestCase):
    def test_pinned_true_pretrained_and_strict_cross_family_role(self):
        self.assertEqual(len(PIN), 40)
        self.assertEqual(BRANCH, "ps/h286-relative-train-label-alignment")
        self.assertEqual(SHIFT_BLOB,
                         "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac")
        self.assertEqual(QUEUE_BLOB,
                         "3cf452e84a75b017905bc14b04a8571411929891")
        self.assertEqual(MODEL, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REV), 40)
        self.assertEqual(STEP_INDICES, (0, 3, 6, 9, 12, 15, 18, 21))
        self.assertEqual(N_STEPS, 8)
        self.assertIn("QUEUE_RESEARCH_NOT_ADMITTED_NOT_OOD", STATUS)
        self.assertEqual(REPORT,
                         "hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json")

    def test_exact_train_and_eval_relative_prompt_same_token_frame(self):
        shifted = model_prompt(
            "Complete fictional queue item S-401" + RELATIVE_SUFFIX)
        self.assertTrue(shifted.endswith(
            RELATIVE_SUFFIX.replace("approval button", "action button")))
        self.assertIn("RELATIVE x,y integers from 0..999", shifted)
        self.assertNotIn("approval button", shifted)
        self.assertIn("correct visible action button", shifted)
        with self.assertRaises(ValueError):
            model_prompt("Complete fictional queue item S-401")
        with self.assertRaises(ValueError):
            model_prompt(None)
        with self.assertRaises(ValueError):
            queue_task(("unregistered", "en", True, "Q-118"))
        self.assertEqual(queue_task(QUEUE_SCENES[1]),
                         "أكمل الطلب الخيالي Q-142")

    def test_unadmitted_train_blocked_if_24_original_source_proofs_missing(self):
        doc = dict(
            role=ROLE, original_source_blob=SHIFT_BLOB,
            relative_targets_roundtrip_verified=24,
            actual_browser_supervisions=24,
            english_candidates=12, arabic_candidates=12,
            real_model_generations=0, official_train_admitted=False,
            source_disjoint_blind_ood=False,
        )
        validate_training_evidence(doc, 24, 12, 12)
        for key, value in (
            ("official_train_admitted", True),
            ("source_disjoint_blind_ood", True),
            ("original_source_blob", "0" * 40),
            ("relative_targets_roundtrip_verified", 0),
        ):
            modified = dict(doc, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_evidence(modified, 24, 12, 12)
        with self.assertRaises(ValueError):
            validate_training_evidence(doc, 8, 4, 4)

    def test_strict_private_t4_cannot_pass_local_windows_gpu_stub(self):
        class NeverCuda:
            @staticmethod
            def is_available():
                return False
        class Stub:
            cuda = NeverCuda()
        with self.assertRaises(ValueError):
            verified_free_t4(Stub())

    def test_full_synthetic_paired_genuine_evidence_contract_no_gain_claim(self):
        doc = verified_report(**verified_kwargs())
        self.assertEqual(doc["paired_pretrained_successes"], 1)
        self.assertEqual(doc["paired_adapted_successes"], 2)
        self.assertEqual(doc["paired_delta"], 1)
        self.assertTrue(doc["same_exact_eval_screens_before_after"])
        self.assertTrue(doc["train_eval_exact_png_disjoint"])
        for flag in ("official_train_dev_ood_admitted",
                     "generalizable_hala_improvement_proven",
                     "adapter_weights_published",
                     "raw_predictions_images_points_or_gold_published"):
            self.assertIs(doc[flag], False)
        public = json.dumps(doc)
        for secret in ('"image"', '"point"', '"gold"', '"prompt"',
                       '"labels"', '"weights"'):
            self.assertNotIn(secret, public)

    def test_fail_closed_fake_optimizer_steps_and_finite_gradients(self):
        for mutation in (
            ("changed", 0), ("changed", True),
            ("coordinates_supervised", 1),
            ("loss_finite", False), ("gradient_nonzero", False),
        ):
            kwargs = verified_kwargs()
            kwargs["steps"][0][mutation[0]] = mutation[1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verified_report(**kwargs)
        kwargs = verified_kwargs()
        kwargs["steps"] = kwargs["steps"][:-1]
        with self.assertRaises(ValueError):
            verified_report(**kwargs)

    def test_fail_closed_pairstate_fabrication_and_mismatched_png(self):
        for field, val in (
            ("model_generation", 0), ("grammar_call", 0),
            ("task_success", 1), ("kind", "abstain"),
            ("native_click", False), ("strict", False),
        ):
            kwargs = verified_kwargs()
            kwargs["base"][0][field] = val
            with self.subTest(field=field), self.assertRaises(ValueError):
                verified_report(**kwargs)
        bad = verified_kwargs()
        bad["after"] = ("c" * 64, "b" * 64)
        with self.assertRaises(ValueError):
            verified_report(**bad)
        bad = verified_kwargs()
        bad["train_hashes"] = frozenset(
            ["a" * 64] + [str(i).zfill(64) for i in range(23)])
        with self.assertRaises(ValueError):
            verified_report(**bad)
        bad = verified_kwargs()
        bad["base"][0]["raw_relative_model_point"] = [500, 700]
        with self.assertRaises(ValueError):
            verified_report(**bad)

    def test_private_kaggle_payload_title_id_and_zero_media_upload(self):
        self.assertEqual(KERNEL_ID,
                         "abdulazizshehri/hala-h287-private-shiftdesk-lora-t4")
        self.assertEqual(KERNEL_TITLE, "Hala H287 Private ShiftDesk LoRA T4")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = Path(__file__).resolve().parents[1] / (
                "h287/private_shiftdesk_lora.py")
            dest = root / "private_code_only"
            info = build(original, dest)
            self.assertEqual(info["uploaded_user_media_weights_or_gold"], 0)
            self.assertEqual(sorted(p.name for p in dest.iterdir()),
                             ["h287_private_shiftdesk_lora.py",
                              "kernel-metadata.json"])
            meta = json.loads((dest / "kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(meta["id"], KERNEL_ID)
            self.assertEqual(meta["title"], KERNEL_TITLE)
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            for key in ("dataset_sources", "model_sources",
                        "competition_sources", "kernel_sources"):
                self.assertEqual(meta[key], [])
            with self.assertRaises(ValueError):
                build(original, dest)


if __name__ == "__main__":
    unittest.main()
