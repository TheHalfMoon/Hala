"""Prevent H419 same-task/different-JPEG leakage in two old/new DEV sets."""
import unittest
from h419.audit_source_tasks import (
    EXPECTED, prove_pairwise_isolated, LABELSETS,
)

class H419SourceTaskIsolationTests(unittest.TestCase):
    def fresh(self):
        images={}
        tasks={}
        for i,key in enumerate(LABELSETS):
            n=EXPECTED[key]
            images[key]={f"img-{i}-{j:05d}" for j in range(n)}
            tasks[key]={f"task-{i}-{j:05d}" for j in range(n)}
        return images,tasks

    def test_entire_raw_train_30000_new_and_original_96_plus_96(self):
        images,tasks=self.fresh()
        result=prove_pairwise_isolated(images,tasks)
        self.assertEqual(len(result),3)
        for overlap in result.values():
            self.assertEqual(overlap["image_sha_overlap"],0)
            self.assertEqual(overlap["original_source_task_sha_overlap"],0)

    def test_different_jpeg_bytes_same_source_task_rejected(self):
        images,tasks=self.fresh()
        self.assertFalse(images["h419_train_original"]&
                         images["h419_new_dev_original"])
        collision=next(iter(tasks["h419_train_original"]))
        tasks["h419_new_dev_original"].add(collision)
        with self.assertRaisesRegex(ValueError,"underlying-task"):
            prove_pairwise_isolated(images,tasks)

    def test_image_sha_overlap_rejected_even_with_new_task_ids(self):
        images,tasks=self.fresh()
        images["h419_new_dev_original"].pop()
        images["h419_new_dev_original"].add(
            next(iter(images["h419_train_original"])))
        self.assertFalse(tasks["h419_train_original"]&
                         tasks["h419_new_dev_original"])
        with self.assertRaisesRegex(ValueError,"source-image"):
            prove_pairwise_isolated(images,tasks)

    def test_missing_source_group_or_wrong_source_budget_fails(self):
        images,tasks=self.fresh()
        del tasks["h417_old_dev_original"]
        with self.assertRaisesRegex(ValueError,"partition keys"):
            prove_pairwise_isolated(images,tasks)
        images,tasks=self.fresh()
        images["h419_train_original"].pop()
        with self.assertRaisesRegex(ValueError,"split changed"):
            prove_pairwise_isolated(images,tasks)

    def test_original_eval_task_set_must_not_be_empty(self):
        images,tasks=self.fresh()
        tasks["h417_old_dev_original"].clear()
        with self.assertRaisesRegex(ValueError,"source tasks missing"):
            prove_pairwise_isolated(images,tasks)

if __name__=="__main__":
    unittest.main()
