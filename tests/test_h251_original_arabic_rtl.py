"""H251 synthetic source-only bilingual RTL contract tests; no original media."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h251.original_lifeos_rtl import (
    CASES, COMMIT, FILES, KINDS, REPO, STAGES, STATUS,
    action, source_preflight, validate_result,
)

def invented_row(case):
    transitions=tuple(zip(STAGES, STAGES[1:]))
    steps=[]
    for i,(start,end) in enumerate(transitions):
        kind=KINDS[i]
        value=(
            action("click",point=[100+i*25,200+i*20])
            if kind=="click" else
            action("type_text",text=case["query"])
            if kind=="type_text" else
            action("scroll",point=[480,600],dy=480)
        )
        steps.append({
            "instruction":"Find the visible project bookmark.",
            "native_action":value,
            "source_frame":start,"result_frame":end,"research_only":True,
        })
    return {
        "case_id":case["id"],
        "language":case["lang"],
        "actual_html_lang":case["lang"],
        "actual_html_dir":"rtl" if case["lang"]=="ar" else "ltr",
        "original_bookmark_saved":True,
        "native_search_visible":True,
        "native_documentation_visible":True,
        "native_scroll_moved":True,
        "model_calls":0,"optimizer_steps":0,
        "train":False,"dev":False,"ood":False,
        "steps":steps,
        "frames":{
            name:{"png_sha256":f"{i+1:064x}",
                  "rgba_sha256":f"{i+20:064x}"}
            for i,name in enumerate(STAGES)
        },
    }

class H251NativeArabicRTLTests(unittest.TestCase):
    def test_two_distinct_original_directions_from_same_app_family(self):
        self.assertEqual(len(CASES),2)
        self.assertEqual({c["lang"] for c in CASES},{"ar","en"})
        self.assertEqual(len(STAGES),6)
        self.assertEqual(KINDS,("click","click","type_text","click","scroll"))
        for case in CASES:
            with self.subTest(lang=case["lang"]):
                validate_result(invented_row(case),case)

    def test_separate_real_source_pin_without_dataset_admission(self):
        self.assertEqual(len(ADMITTED_TRAIN_MANIFEST_SHAS),0)
        self.assertIn("NOT_ADMITTED",STATUS)
        self.assertEqual(REPO,"3mmar404/life-os-free")
        self.assertEqual(len(COMMIT),40)
        self.assertIn("LICENSE",FILES)
        self.assertIn("index.html",FILES)
        self.assertIn("js/core.js",FILES)

    def test_invalid_state_or_implicit_gold_rejected(self):
        mods=(
            lambda x:x.update(actual_html_lang="en" if x["language"]=="ar" else "ar"),
            lambda x:x.update(actual_html_dir="ltr" if x["language"]=="ar" else "rtl"),
            lambda x:x.update(original_bookmark_saved=False),
            lambda x:x.update(native_search_visible=False),
            lambda x:x.update(native_scroll_moved=False),
            lambda x:x.update(train=True),
            lambda x:x.update(dev=True),
            lambda x:x.update(ood=True),
            lambda x:x.update(model_calls=1),
            lambda x:x.update(optimizer_steps=1),
            lambda x:x["steps"][2].update(source_frame="before_nav"),
            lambda x:x["steps"][2].update(
                native_action=action("type_text",text="wrong text")),
            lambda x:x["steps"][0].update(research_only=False),
            lambda x:x["frames"]["after_type"].update(
                png_sha256=x["frames"]["after_focus"]["png_sha256"]),
            lambda x:x["frames"].pop("after_docs"),
        )
        for mod in mods:
            v=invented_row(CASES[0])
            mod(v)
            with self.subTest(mod=repr(mod)),self.assertRaises(ValueError):
                validate_result(v,CASES[0])

    def test_fake_unfrozen_or_missing_original_repo_rejected(self):
        with TemporaryDirectory() as root:
            with self.assertRaises((FileNotFoundError,ValueError)):
                source_preflight(Path(root)/"never-original",
                                 Path(root)/"h251-owner-private-research-never")

if __name__=="__main__":
    unittest.main()
