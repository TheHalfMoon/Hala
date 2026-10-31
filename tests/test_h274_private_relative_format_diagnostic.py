"""Synthetic-only H274 private format diagnosis and payload guards."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h273.relative_native_bridge import NORMALIZED_REGEX
from h274.private_relative_format_diagnostic import (
    classify_strict_model_text,summarize,SCENES,REPORT)
from h274.build_private_payload import (
    build,KERNEL_ID,SCRIPT_NAME,META_NAME)

class H274PrivateFormatTests(unittest.TestCase):
    def test_invented_format_classes_are_safe_coarse_labels_only(self):
        valid='{"actions":[{"kind":"click","point":[500,500]}]}'
        samples={
            valid:"STRICT_EXACT_VALID",
            '{"actions":[{"kind":"abstain"}]}':"STRICT_EXACT_VALID",
            valid+" ":"JSON_VALID_CLICK_NONCANONICAL_SYNTAX",
            valid[:-2]:"INCOMPLETE_OR_UNCLOSED_JSON",
            '{"actions":[{"kind":"click","point":[1000,500]}]}':
                "JSON_CLICK_INVALID_RELATIVE_POINT",
            '{"actions":[{"kind":"abstain"}],"source":"secret"}':
                "JSON_DIFFERENT_TOP_LEVEL_KEYS",
            "": "EMPTY",
            "not-json":"NOT_JSON_OR_NONCANONICAL_PREFIX",
            (chr(96)*3)+"json\n"+valid:
                "MARKDOWN_FENCE",
            "x"*129:"OVER_128_CHARS",
        }
        for raw,expected in samples.items():
            with self.subTest(expected=expected):
                got=classify_strict_model_text(raw,NORMALIZED_REGEX)
                self.assertEqual(got,expected)
                self.assertNotIn("500",got)
                self.assertNotIn("secret",got)

    def test_diagnostic_refuses_fake_model_or_unscored_benchmark_claim(self):
        rows=[{"scene":s,"real_model_generation_count":1,
               "real_token_filter_generate_calls":1,
               "original_screenshot_unchanged":True,
               "format_class":"INCOMPLETE_OR_UNCLOSED_JSON"}
              for s in SCENES]
        report=summarize(rows,{"free_t4":True,"devices":2})
        self.assertEqual(report["real_pretrained_image_generations"],2)
        self.assertFalse(report["official_train_dev_ood_admitted"])
        self.assertFalse(report["independent_blind_ood"])
        self.assertFalse(report["generalizable_hala_skill_proven"])
        self.assertFalse(report["raw_model_text_points_or_images_saved"])
        self.assertFalse(report["weights_or_adapters_published"])
        self.assertEqual(report["native_actions_dispatched"],0)
        self.assertNotIn('"raw_output":',json.dumps(report))
        rows[1]["real_model_generation_count"]=0
        with self.assertRaises(ValueError):
            summarize(rows,{"free_t4":True})
        rows[1]["real_model_generation_count"]=1
        with self.assertRaises(ValueError):
            summarize(rows,{"free_t4":False})
        self.assertEqual(REPORT,
          "hala-h274-private-relative-format-reason-aggregate.json")

    def test_private_code_only_onefile_payload(self):
        root=Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as td:
            path=Path(td)/"new"
            receipt=build(root/"h274"/"private_relative_format_diagnostic.py",
                          path)
            self.assertTrue(receipt["private"])
            self.assertEqual(receipt["uploaded_model_or_media"],0)
            self.assertEqual({p.name for p in path.iterdir()},
                             {SCRIPT_NAME,META_NAME})
            info=json.loads((path/META_NAME).read_text("utf-8"))
            script=(path/SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(info["id"],KERNEL_ID)
            self.assertEqual(info["is_private"],"true")
            self.assertEqual(info["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(info["dataset_sources"],[])
            self.assertEqual(info["model_sources"],[])
            compile(script,"private_h274_merged","exec")
            self.assertIn("/tmp/hala-h274-private-format-reasons",script)
            self.assertIn('page.locator("#search").evaluate("(el) => el.blur()")',script)
            self.assertLess(script.index('page.locator("#search").evaluate'),
                            script.index("raw=policy(png,RELATIVE_PROMPT)"))
            self.assertIn("png=_observed_png(page)",script)
            self.assertIn("original_screenshot_unchanged",script)
            self.assertNotIn('HF_HOME="/kaggle/working',script)
            with self.assertRaises(ValueError):
                build(root/"h274"/"private_relative_format_diagnostic.py",
                      path)

if __name__=="__main__":
    unittest.main()
