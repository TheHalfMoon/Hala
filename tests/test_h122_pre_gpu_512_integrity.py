"""H122 CPU pre-GPU checks: authentic image bytes, TRAIN/heldout separation.

No GPU, model download, EVAL, or owner-private files are needed in CI.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from h122.pre_gpu import qualify_before_gpu
from h122.schedule import Schedule,TRAIN,HOLDOUT,ORIGINAL
from h122.training_plan import Plan


def fixture(root):
    (root/"images").mkdir()
    sources={}
    rows=[]
    for i in range(ORIGINAL):
        raw=f"original independent TRAIN image #{i}".encode("utf8")
        image=f"images/screenshot-{i:03}.jpg"
        (root/image).write_bytes(raw)
        ident=f"original-train-{i:03}"
        row={"id":ident,"split":"train","image_path":image,
             "image_sha256":hashlib.sha256(raw).hexdigest(),
             "family":f"family-{i%5}",
             "language":"ar" if i%4==0 else "en"}
        rows.append(row)
        sources[ident]=dict(row)
    train_idx=tuple(range(TRAIN))
    hold_idx=tuple(range(TRAIN,ORIGINAL))
    report=lambda indices,field:dict(Counter(rows[i][field] for i in indices))
    schedule=Schedule(
        train=tuple({"original_source_id":rows[i]["id"]} for i in train_idx),
        heldout=tuple({"original_source_id":rows[i]["id"]} for i in hold_idx),
        train_original_indices=train_idx,
        heldout_original_indices=hold_idx,
        summary={"source_split":"TRAIN_ONLY","no_EVAL_or_TEST_source":True,
                 "no_holdout_optimizer_input":True,"schedule_sha256":"frozen",
                 "train_families":report(train_idx,"family"),
                 "heldout_families":report(hold_idx,"family"),
                 "train_languages":report(train_idx,"language"),
                 "heldout_languages":report(hold_idx,"language")})
    plan=Plan(indices=train_idx,sha256="train-448",
              summary={"original_full512_train_sha256":"parent",
                       "h122_schedule_sha256":"frozen"})
    return rows,sources,schedule,plan


class OriginalSourceBeforeGPU(unittest.TestCase):
    def run_with_source(self,root,rows,sources,schedule,plan):
        with patch("h122.pre_gpu.full_original_plan",
                   return_value=SimpleNamespace(indices=tuple(range(ORIGINAL)),
                                                sha256="parent")),\
             patch("h122.pre_gpu.admit_original_source_map",
                   return_value=sources),\
             patch("h122.pre_gpu.original_for_sft",
                   side_effect=lambda row,all_sources:all_sources[row["id"]]),\
             patch("h122.pre_gpu.verify_overlay",return_value=True):
            return qualify_before_gpu(rows,root,schedule,plan)

    def test_512_original_train_images_qualified_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows,sources,schedule,plan=fixture(root)
            report=self.run_with_source(root,rows,sources,schedule,plan)
            self.assertEqual(report["original_source_images_sha_verified"],512)
            self.assertEqual(report["optimizer_train_images_sha_verified"],448)
            self.assertEqual(report["excluded_heldout_train_images_sha_verified"],64)
            self.assertFalse(report["model_initialized_or_trained"])
            self.assertTrue(report["no_heldout_optimizer_input"])
            self.assertNotIn("ids",report)

    def test_modified_heldout_bytes_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows,sources,schedule,plan=fixture(root)
            (root/rows[500]["image_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"screenshot bytes differ"):
                self.run_with_source(root,rows,sources,schedule,plan)

    def test_optimizer_excluding_real_training_image_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows,sources,schedule,plan=fixture(root)
            swapped=Plan(indices=tuple(range(447))+(500,),
                         sha256=plan.sha256,summary=plan.summary)
            with self.assertRaisesRegex(ValueError,"nonoriginal or heldout"):
                self.run_with_source(root,rows,sources,schedule,swapped)

    def test_train_and_heldout_disjoint_partition_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows,sources,schedule,plan=fixture(root)
            forged=Schedule(train=schedule.train,heldout=schedule.heldout,
                train_original_indices=schedule.train_original_indices,
                heldout_original_indices=tuple(range(700,764)),
                summary=schedule.summary)
            with self.assertRaisesRegex(ValueError,"448/64"):
                self.run_with_source(root,rows,sources,forged,plan)


if __name__=="__main__":
    unittest.main()
