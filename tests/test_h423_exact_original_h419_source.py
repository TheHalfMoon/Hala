"""H423 guarantee SAME original H419 source order and native loss in rescue.

A resumed model is meaningful only if exact genuine H419 sampled images,
labels, source SHA, native two-point JSON and numeric-weighted loss remain
identical. These are source-code equivalence assertions, NOT GPU progress.
"""
import inspect
import unittest

from h419.kaggle import h419_native_drag_continuation as original
from h423.kaggle import h423_h419_checkpoint_resume as rescue


class H423OriginalSourceEquivalence(unittest.TestCase):
    def test_all_original_source_sha_and_revisions_identical(self):
        for name in (
            "BASE","REV","SOURCE_REV","DATASET_SHA",
            "FIRSTPARTY_ZIP_SHA","FIRSTPARTY_LABEL_SHA",
            "SOURCE_SHAS","BIL_FILE","CURRIC_FILE",
            "DOMAINS","H419_EXTRA_SHAS","H419_ALL_SOURCE_SHAS",
            "H418_ROWS_SHA","H418_RECEIPT_SHA",
            "PARENT_RECEIPT_SHA","PARENT_ADAPTER_SHA",
            "H419_TRAINING_PASSES","H419_STEPS","H419_SEED",
            "H419_AR_N","H419_NEW_IMAGE_N",
            "H419_AR_ADDITIONAL_PASSES","H419_DEV_N",
            "GRAD_ACCUM","H419_OLD_DEV_N","H419_NEW_DEV_N",
        ):
            with self.subTest(source=name):
                self.assertEqual(getattr(original,name),getattr(rescue,name))

    def test_genuine_h419_partition_and_random_shuffle_byte_identical(self):
        for name in (
            "get_research_sources","case_from_dragon",
            "case_from_firstparty","partition",
            "open_extra_sources","new_verified_images",
            "new_source_development","h419_prepare_sources",
        ):
            with self.subTest(function=name):
                self.assertEqual(
                    inspect.getsource(getattr(original,name)),
                    inspect.getsource(getattr(rescue,name)))

    def test_true_H419_input_label_and_ordered_endpoint_loss_same(self):
        for name in (
            "sha_bytes","sha_file","strict_load",
            "source_image","target","prompt",
            "to_inputs","coordinate_weighted_loss",
            "score_predicted","evaluate",
        ):
            with self.subTest(function=name):
                self.assertEqual(
                    inspect.getsource(getattr(original,name)),
                    inspect.getsource(getattr(rescue,name)))

    def test_resume_records_explicit_adamw_reset_not_full_original_qualification(self):
        source=inspect.getsource(rescue.run_h423)
        self.assertIn('"uninterrupted_9452_step_optimizer_history":False',source)
        self.assertIn('"optimizer_moments_restored_from_H419":False',source)
        self.assertIn('"AdamW_optimizer_reset_on_resume":True',source)
        self.assertIn('order[absolute_step*GRAD_ACCUM+micro]',source)
        self.assertNotIn("source_only_research_gate=True",source)
        self.assertNotIn("model_release_qualified=True",source)
        self.assertEqual(rescue.H423_T4_SEGMENT_STEPS,1500)

if __name__=="__main__":
    unittest.main()
