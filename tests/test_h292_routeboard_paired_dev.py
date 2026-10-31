"""H292 strict synthetic source/receipt regression — NO GPU or real model.

These invented report values are not Qwen outcomes. A real owner-private
FREE T4 terminal run and exact whitelisted receipt are separately required.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from h280.verify_private_native_aggregate import EvidenceRejected
from h292.private_paired_routeboard_dev import (
    COMPOUND_REPORT, COMPOUND_STATUS, DEV_SOURCE_BRANCH,
    DEV_SOURCE_SHA, DEV_SOURCE_BLOB, DEV_INDICES, FLAG,
    MODEL, MODEL_REV, STEP_INDICES, N_STEPS, QUEUE_SCENES,
    verified_report, failure_receipt,
)
from h292.verify_paired_dev_receipt import (
    verify_bytes, verify_private_file, DEV_SCENES, DEV_LANGUAGES,
)
from h292.verify_stage_diagnostic import verify_bytes as verify_failure
from h292.build_private_payload import build, KERNEL_ID, KERNEL_TITLE
from h289.routeboard_dev_research import (
    SCENES as ROUTE_SCENES, BLOB as ORIGINAL_DEV_BLOB,
)


def original_rows(successes):
    return [{
        "scene": scene[0], "model_generation": 1, "grammar_call": 1,
        "strict": True, "kind": "click", "native_click": True,
        "task_success": bool(successes[i]),
    } for i, scene in enumerate(QUEUE_SCENES)]


def dev_rows(successes):
    return [{
        "scene": DEV_SCENES[i], "language": DEV_LANGUAGES[i],
        "model_generation": 1, "grammar_call": 1,
        "strict": True, "kind": "click", "native_click": True,
        "task_success": bool(successes[i]),
    } for i in range(6)]


def forged_report(base=(True,False,False,True,False,False),
                  adapted=(True,True,False,True,False,True)):
    queue = verified_report(
        steps=[{
            "changed": 3, "coordinates_supervised": True,
            "loss_finite": True, "gradient_nonzero": True,
        } for _ in range(N_STEPS)],
        base=original_rows((True, False)),
        adapted=original_rows((True, False)),
        before=("a"*64,"b"*64), after=("a"*64,"b"*64),
        platform={"free_t4": True, "devices": 1},
        train_hashes=frozenset(str(i).zfill(64) for i in range(24)),
    )
    return {
        "status": COMPOUND_STATUS,
        "original_queueboard_research": queue,
        "original_dev_source_commit": DEV_SOURCE_SHA,
        "original_dev_html_blob": DEV_SOURCE_BLOB,
        "original_dev_exposed_not_blind_ood": True,
        "dev_scenes_preregistered": 6,
        "dev_model_frames_distinct_and_train_queue_disjoint": True,
        "paired_dev_screenshots_identical": True,
        "dev_base": dev_rows(base),
        "dev_adapted": dev_rows(adapted),
        "dev_base_successes": sum(base),
        "dev_adapted_successes": sum(adapted),
        "dev_paired_delta": sum(adapted)-sum(base),
        "official_train_dev_ood_admitted": False,
        "generalized_hala_skill_proven": False,
        "raw_predictions_images_points_gold_or_weights_saved": False,
    }


def data(doc):
    return json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()


class H292PairedOriginalThreeGUIResearch(TestCase):
    def test_prospective_immutable_firstparty_dev_and_true_training(self):
        self.assertEqual(DEV_SOURCE_BRANCH, "ps/h289-original-routeboard-dev-research")
        self.assertEqual(DEV_SOURCE_SHA,
                         "5a7608f533897ac9c10d41b3d97c4a0b245a000b")
        self.assertEqual(DEV_SOURCE_BLOB, ORIGINAL_DEV_BLOB)
        self.assertEqual(DEV_INDICES, (0,1,4,6,7,10))
        chosen = tuple(ROUTE_SCENES[i] for i in DEV_INDICES)
        self.assertEqual(tuple(s[0] for s in chosen), DEV_SCENES)
        self.assertEqual(tuple(s[1] for s in chosen), DEV_LANGUAGES)
        self.assertEqual(N_STEPS, len(STEP_INDICES) == 8 and 8)
        self.assertEqual(MODEL, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REV), 40)
        self.assertEqual(FLAG, "HALA_H292_PRIVATE_NONCOMMERCIAL_FREE_T4")

    def test_synthetic_source_emitter_to_closed_independent_compound(self):
        safe = verify_bytes(data(forged_report()))
        self.assertEqual(safe["verified_research_optimizer_steps"], 8)
        self.assertEqual(safe["source_train_candidate_count"], 24)
        self.assertEqual(safe["original_queueboard"]["base"]["task_successes"],1)
        self.assertEqual(safe["original_queueboard"]["adapted"]["task_successes"],1)
        self.assertEqual(safe["queueboard_paired_delta"],0)
        self.assertEqual(safe["original_routeboard"]["base"]["task_successes"],2)
        self.assertEqual(safe["original_routeboard"]["adapted"]["task_successes"],4)
        self.assertEqual(safe["routeboard_paired_delta"],2)
        self.assertFalse(safe["blind_ood_proven"])
        self.assertFalse(safe["official_train_dev_admitted"])
        self.assertFalse(safe["generalizable_gain_proven"])
        text = json.dumps(safe)
        for secret in ('"point"','"image"','"gold"','"prompt"','"weights"'):
            self.assertNotIn(secret,text)

    def test_negative_zero_and_positive_dev_deltas_honest(self):
        for base, adapted, delta in (
            ((True,)*6,(False,)*6,-6),
            ((True,False)*3,(True,False)*3,0),
            ((False,)*6,(True,)*6,6),
        ):
            with self.subTest(delta=delta):
                receipt=forged_report(base,adapted)
                self.assertEqual(verify_bytes(data(receipt))["routeboard_paired_delta"],delta)

    def test_reject_fabricated_state_native_and_raw_private_fields(self):
        wrongs = (
            ("model_generation",0), ("grammar_call",0), ("strict",1),
            ("native_click",0), ("task_success",1),
            ("kind","abstain"), ("scene","dev_wrong"),
            ("language","ar"),
        )
        for key,value in wrongs:
            doc=forged_report()
            doc["dev_base"][0][key]=value
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_bytes(data(doc))
        for loc in ("root","devrow","nestedoriginal"):
            doc=forged_report()
            t=doc if loc=="root" else (
                doc["dev_adapted"][0] if loc=="devrow" else
                doc["original_queueboard_research"])
            t["private_gold_coordinates"]=[12,34]
            with self.subTest(loc=loc),self.assertRaises(EvidenceRejected):
                verify_bytes(data(doc))

    def test_reject_false_provenance_admission_and_count(self):
        for key,value in (
            ("original_dev_source_commit","0"*40),
            ("original_dev_html_blob","0"*40),
            ("original_dev_exposed_not_blind_ood",False),
            ("dev_model_frames_distinct_and_train_queue_disjoint",False),
            ("paired_dev_screenshots_identical",False),
            ("dev_scenes_preregistered",4),
            ("dev_paired_delta",0),
            ("dev_adapted_successes",0),
            ("official_train_dev_ood_admitted",True),
            ("generalized_hala_skill_proven",True),
            ("raw_predictions_images_points_gold_or_weights_saved",True),
        ):
            doc=forged_report()
            doc[key]=value
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_bytes(data(doc))
        doc=forged_report()
        doc["original_queueboard_research"]["research_train_optimizer_steps"]=0
        with self.assertRaises(EvidenceRejected):
            verify_bytes(data(doc))

    def test_reject_duplicate_encoded_keys_nan_invalid_and_oversize(self):
        original=data(forged_report())
        for tampered in (
            original.replace(b'"status":',b'"\\u0073tatus":"bad","status":',1),
            original.replace(b'"dev_paired_delta":2',b'"dev_paired_delta":NaN',1),
            b"\xff"*250,b"NOT_JSON",b" "*8501,
        ):
            with self.subTest(bytes=len(tampered)),self.assertRaises(EvidenceRejected):
                verify_bytes(tampered)

    def test_private_exact_filename_and_bounded_report(self):
        with TemporaryDirectory() as p:
            root=Path(p)
            good=root/COMPOUND_REPORT
            raw=data(forged_report())
            self.assertLessEqual(len(raw),8500)
            good.write_bytes(raw)
            self.assertEqual(verify_private_file(good)["routeboard_paired_delta"],2)
            wrong=root/"unfiltered.json"
            wrong.write_bytes(raw)
            with self.assertRaises(EvidenceRejected):
                verify_private_file(wrong)
            try:
                link=root/"private-alias.json"
                link.symlink_to(good)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_private_file(link)

    def test_stage_only_failure_parser_no_model_gain(self):
        receipt=failure_receipt("DEV_ADAPTED","ValueError")
        safe=verify_failure(data(receipt))
        self.assertEqual(safe["failure_stage"],"DEV_ADAPTED")
        self.assertEqual(safe["verified_optimizer_steps"],0)
        self.assertFalse(safe["verified_paired_gui_scores"])

    def test_code_only_package_valid_generated_python_without_media(self):
        src=Path(__file__).resolve().parents[1]/"h292/private_paired_routeboard_dev.py"
        with TemporaryDirectory() as folder:
            path=Path(folder)/"private"
            m=build(src,path)
            self.assertEqual(m["uploaded_media_weights_or_gold"],0)
            self.assertEqual(sorted(x.name for x in path.iterdir()),[
                "h292_private_paired_routeboard_dev.py","kernel-metadata.json",
            ])
            metadata=json.loads((path/"kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["id"],KERNEL_ID)
            self.assertEqual(metadata["title"],KERNEL_TITLE)
            self.assertEqual(metadata["model_sources"],[])
            self.assertEqual(metadata["dataset_sources"],[])
            ast.parse((path/"h292_private_paired_routeboard_dev.py").read_text("utf-8"))
            with self.assertRaises(ValueError):
                build(src,path)
