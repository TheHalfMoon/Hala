"""CPU-only H140 neural receipt and strict original-DEV anti-collapse tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h140.train_gpu import (
    STEPS,GRAD_ACCUM,TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA,
    EXPECTED_APP_SET,summarize_native,private_splits,
)
from h140.qualification import SCHEMA,WIDTH,HEIGHT
from h132.grounder_contract import native_point
from h140.train_gpu_v3 import summarize_native as summarize_native_v3


class H140NativeGUIResearchGrounderContract(unittest.TestCase):
    @staticmethod
    def cases():
        sizes={"vanilla-native-board":22,
               "react-native-board":22,
               "react-dndkit-board":20}
        rows=[]
        for app,n in sizes.items():
            for i in range(n):
                x=40+(i%16)*26
                rows.append({"id":f"{app}-{i}","app":app,
                             "language":"ar" if i%2 else "en",
                             "start_bbox":{"x":x,"y":160,"width":20,"height":20},
                             "end_bbox":{"x":x+40,"y":360,"width":20,"height":20}})
        return {"cases":rows}

    @staticmethod
    def raw_for_point(point):
        grid=[round(point[0]*1000/WIDTH),round(point[1]*1000/HEIGHT)]
        raw=json.dumps({"point_2d":grid},separators=(",",":"))
        if native_point(raw,WIDTH,HEIGHT)!=point:
            raise AssertionError("H390 test point must roundtrip native parser")
        return raw

    @classmethod
    def predictions(cls,doc,collapsed=False):
        out=[]
        for row in doc["cases"]:
            for role in ("start","end"):
                rect=row[role+"_bbox"]
                point=[42,44] if collapsed else [
                    int(rect["x"]+rect["width"]/2),
                    int(rect["y"]+rect["height"]/2)]
                hit=(rect["x"]<=point[0]<=rect["x"]+rect["width"] and
                     rect["y"]<=point[1]<=rect["y"]+rect["height"])
                out.append({"case_id":row["id"],"role":role,
                            "app":row["app"],"language":row["language"],
                            "raw":cls.raw_for_point(point),
                            "point":point,"hit":bool(hit)})
        return out

    def test_balanced_one_epoch_all_512_source_roles(self):
        self.assertEqual((STEPS,GRAD_ACCUM),(128,4))
        self.assertEqual(STEPS*GRAD_ACCUM,512)
        self.assertEqual(len(TRAIN_MANIFEST_SHA),64)
        self.assertEqual(len(DEV_MANIFEST_SHA),64)
        self.assertEqual(len(EXPECTED_APP_SET),3)

    def test_diverse_all_three_apps_and_both_roles_pass(self):
        doc=self.cases()
        result=summarize_native(self.predictions(doc),doc)
        self.assertTrue(result["all_apps_and_roles_diverse"])
        self.assertEqual(result["point_hits"]["__all__"],
                         {"start":64,"end":64,"both":64,"cases":64})
        for roles in result["point_diversity_by_app"].values():
            self.assertGreaterEqual(roles["start"]["unique_points"],8)
            self.assertGreaterEqual(roles["end"]["unique_points"],8)

    def test_fixed_native_points_fail_even_if_model_text_is_valid(self):
        doc=self.cases()
        result=summarize_native(self.predictions(doc,collapsed=True),doc)
        self.assertFalse(result["all_apps_and_roles_diverse"])
        for roles in result["point_diversity_by_app"].values():
            self.assertFalse(roles["start"]["passed"])
            self.assertFalse(roles["end"]["passed"])
        with self.assertRaises(ValueError):
            summarize_native(self.predictions(doc)[:-1],doc)

    def test_v1_v3_native_summaries_refuse_subpixel_jitter(self):
        doc=self.cases()
        base_rows=self.predictions(doc)
        truth={row["id"]:row for row in doc["cases"]}
        for item in base_rows:
            case_index=int(item["case_id"].rsplit("-",1)[1])
            point=[96+case_index%16,201]
            rect=truth[item["case_id"]][item["role"]+"_bbox"]
            item["point"]=point
            item["raw"]=self.raw_for_point(point)
            item["hit"]=(rect["x"]<=point[0]<=rect["x"]+rect["width"] and
                         rect["y"]<=point[1]<=rect["y"]+rect["height"])
        for summarize in (summarize_native,summarize_native_v3):
            with self.subTest(summarizer=summarize.__module__):
                result=summarize(base_rows,doc)
                self.assertFalse(result["all_apps_and_roles_diverse"])
                for roles in result["point_diversity_by_app"].values():
                    for role in ("start","end"):
                        self.assertGreaterEqual(roles[role]["unique_points"],8)
                        self.assertEqual(roles[role]["unique_position_bins"],1)
                        self.assertFalse(roles[role]["passed"])

    def test_v1_v3_reject_unbalanced_language_in_selected_geometry_slice(self):
        doc=self.cases()
        groups={}
        for row in doc["cases"]:
            groups.setdefault(row["app"],[]).append(row)
        # Whole DEV remains EN/AR balanced; its first 16 per app are not.
        doc["cases"]=[
            row for app in sorted(groups)
            for row in sorted(groups[app],
                              key=lambda r: (r["language"]!="en",r["id"]))
        ]
        for summarize in (summarize_native,summarize_native_v3):
            with self.subTest(summarizer=summarize.__module__):
                with self.assertRaisesRegex(ValueError,"balanced EN/AR"):
                    summarize(self.predictions(doc),doc)

    def test_v1_v3_reject_arabic_only_position_collapse(self):
        doc=self.cases()
        by_id={row["id"]:row for row in doc["cases"]}
        responses=self.predictions(doc)
        for item in responses:
            row=by_id[item["case_id"]]
            if row["language"]=="ar":
                index=int(item["case_id"].rsplit("-",1)[1])
                group=(index//2)%2
                point=[600+50*group+index%8,500]
                item["point"]=point
                item["raw"]=self.raw_for_point(point)
                item["hit"]=False
        for summarize in (summarize_native,summarize_native_v3):
            with self.subTest(summarizer=summarize.__module__):
                result=summarize(responses,doc)
                self.assertFalse(result["all_apps_and_roles_diverse"])
                for roles in result["point_diversity_by_app"].values():
                    for role in ("start","end"):
                        summary=roles[role]
                        self.assertGreaterEqual(summary["unique_position_bins"],8)
                        self.assertLessEqual(summary["largest_position_bin_count"],4)
                        self.assertTrue(summary["bilingual_spatial"]["en"]["passed"])
                        self.assertFalse(summary["bilingual_spatial"]["ar"]["passed"])
                        self.assertEqual(summary["bilingual_spatial"]["ar"]["unique_position_bins"],2)

    def test_v1_v3_refuse_unattested_native_receipt_fields(self):
        """Even a correct 16-case diversity slice cannot excuse forged last-row data."""
        doc=self.cases()
        for corruption in ("unparseable_raw","point_raw_mismatch","wrong_app",
                           "wrong_language","nonboolean_hit","missing_raw",
                           "nan_point_outside_slice"):
            for summarize in (summarize_native,summarize_native_v3):
                with self.subTest(kind=corruption,summarizer=summarize.__module__):
                    records=self.predictions(doc)
                    item=records[-1]  # Last row is NOT in the first 16 per app.
                    if corruption=="unparseable_raw":
                        item["raw"]="synthetic prose is not model-native JSON"
                    elif corruption=="point_raw_mismatch":
                        item["point"][0]+=1  # still inside ground truth bbox
                    elif corruption=="wrong_app":
                        item["app"]="vanilla-native-board"
                    elif corruption=="wrong_language":
                        item["language"]="en"
                    elif corruption=="nonboolean_hit":
                        item["hit"]=1
                    elif corruption=="missing_raw":
                        del item["raw"]
                    elif corruption=="nan_point_outside_slice":
                        item["point"]=[float("nan"),100]
                        item["hit"]=False
                    with self.assertRaisesRegex(ValueError,"H140"):
                        summarize(records,doc)

    def test_native_receipt_invalid_raw_can_only_abstain(self):
        doc=self.cases()
        for summarize in (summarize_native,summarize_native_v3):
            with self.subTest(summarizer=summarize.__module__):
                records=self.predictions(doc)
                records[-1]["raw"]="NOT_JSON_ABORT"
                records[-1]["point"]=None
                records[-1]["hit"]=False
                result=summarize(records,doc)
                self.assertEqual(result["point_hits"]["__all__"]["end"],63)
                self.assertEqual(result["point_hits"]["__all__"]["cases"],64)

    def test_training_mount_refuses_even_single_blind_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            blind=root/"sealed-model-blind"
            blind.mkdir()
            (blind/"manifest.json").write_text(json.dumps({
                "schema":SCHEMA,"split":"blind_test","cases":[]}))
            # Exercise the independent BLIND-mount rejection past the new
            # H199 rights gate using a test-only synthetic registry patch.
            from h199 import native_action_sft as rights_contract
            with (patch("h140.train_gpu.DATA_ROOT",root),
                  patch.object(rights_contract,"ADMITTED_TRAIN_MANIFEST_SHAS",
                               frozenset({TRAIN_MANIFEST_SHA}))):
                with self.assertRaisesRegex(ValueError,"refuse BLIND mount"):
                    private_splits()


if __name__=="__main__":
    unittest.main()
