"""Owner-local source contract for H412. No GPU, external screenshots or secrets."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

from h412.kaggle import h412_duplicate_curriculum_lora as candidate


class H412CurriculumTests(unittest.TestCase):
    def test_frozen_reweighting_and_model_lineage(self):
        self.assertEqual((candidate.TRAIN_N, candidate.DEV_N, candidate.AUDIT_N, candidate.STEPS), (192, 48, 48, 192))
        receipt = Path(__file__).resolve().parents[1] / "h409" / "verified_receipt" / "h409-owner-private-balanced-ui-lora" / "receipt.json"
        original = receipt.read_bytes()
        self.assertEqual(sha256(original).hexdigest(), candidate.PARENT_RECEIPT_SHA)
        state = json.loads(original)
        self.assertEqual(state["adapter_sha256"], candidate.PARENT_SHA)
        self.assertEqual(state["status"], "COMPLETE_PRIVATE_TWO_FAMILY_128_STEP_LORA")
        self.assertEqual(state["optimizer_steps"], 128)
        self.assertIs(state["weights_public"], False)
        self.assertIs(state["official_hala_model_trained"], False)
        import inspect
        self.assertIn('r.get("official_hala_model_trained")', inspect.getsource(candidate.private_parent))

    def test_private_free_gpu_input_config(self):
        metadata = json.loads((Path(__file__).resolve().parents[1] / "h412" / "kaggle" / "kernel-metadata.json").read_text())
        self.assertIs(metadata["is_private"], True)
        self.assertIs(metadata["enable_gpu"], True)
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["kernel_sources"], ["abdulazizshehri/hala-h409-private-balanced-original-ui-lora"])
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertEqual(metadata["model_sources"], [])

    def test_split_design_and_train_contrast(self):
        counts = Counter()
        for split, count in (("train",192),("dev",48),("audit",48)):
            for i in range(count):
                family,kind,j = candidate.pick_family_kind(i,split)
                counts[(split,family,kind,j%2)] += 1
            for family in ("button_ui","color_grid"):
                for kind in candidate.KINDS:
                    want = (16 if kind=="unique" else 8 if kind=="absent" else 24) if split=="train" else 4
                    for lang in (0,1):
                        self.assertEqual(counts[(split,family,kind,lang)],want)

    def test_source_pixels_and_prompts_nonoverlap_previous_h409(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional for no-site CI")
        from h409.kaggle import h409_balanced_two_family_lora as parent
        old = {
            sha256(parent.compose(i,seed,split)[0].tobytes()).hexdigest()
            for split,seed,count in (("train",40911,128),("dev",40921,48),("audit",40931,48))
            for i in range(count)
        }
        source = {}
        for split,seed,count in (("train",41211,192),("dev",41221,48),("audit",41231,48)):
            rows=[candidate.compose(i,seed,split) for i in range(count)]
            source[split]=rows
            self.assertTrue(all(len(boxes)=={"unique":1,"absent":0,"ambiguous":2}[kind]
                                for image,prompt,gold,family,kind,lang,boxes in rows))
            pix={sha256(image.tobytes()).hexdigest() for image,*_ in rows}
            self.assertEqual(len(pix),count)
            self.assertFalse(pix & old,"H412 reused previous H409 source or audit pixel")
        hashes=candidate.freeze(source["train"],source["dev"],source["audit"])
        self.assertEqual(set(hashes),{"train","dev","audit"})
        self.assertEqual(len(set(hashes.values())),3)


if __name__ == "__main__":
    unittest.main()
