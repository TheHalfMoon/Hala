"""H244 cross-cohort private screenshot integrity: synthetic only tests.

No old image bytes/gold, no H173 OOD or model weights. All image fixtures
are invented PIL images in OS temp, and private receipts never uploaded.
"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from h204.training_readiness_gap import training_readiness_gap
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import COMMIT,PINNED,REPO
from h242.private_dragula_prospects import CASES
from h244.private_research_overlap import (
    audit_owner_research_overlap,historical_h236_research,verify_h242_receipt,
)
from test_h204_training_readiness_gap import fixtures


def sample_data():
    data={
        "status":"H242_OWNER_PRIVATE_ORIGINAL_DRAGULA_PROSPECTS_NOT_ADMITTED",
        "source_repository":REPO,"source_git_sha":COMMIT,
        "source_asset_sha256":dict(PINNED),
        "original_js_runtime_unmodified":True,"rendered_html_self_authored":True,
        "official_train_dev_dataset_admitted":False,"train_eligible":False,
        "dev_eligible":False,"ood_eligible":False,
        "train_optimizer_approved":False,
        "full_historical_source_and_visual_history_proven":False,
        "transitive_license_and_media_rights_audited":False,
        "training_model_calls":0,"optimizer_steps":0,"cases":[],
    }
    blobs={}
    for i,s in enumerate(CASES):
        item={
            "case_id":s["case_id"],"language":s["lang"],
            "instruction":s["task"],"source_point":[180,175],
            "destination_point":[550,240],
            "native_drag_verified":True,"network_requests":0,
            "training_eligible":False,"dev_eligible":False,
            "ood_eligible":False,
        }
        for phase_idx,stage in enumerate(("before","after")):
            rgb=((i*25+phase_idx*4+17)%256,
                 (i*24+phase_idx*6+33)%256,
                 (i*23+phase_idx*8+58)%256)
            image=Image.new("RGB",(960,720),rgb)
            buffer=BytesIO()
            image.save(buffer,format="PNG")
            png=buffer.getvalue()
            key="preaction" if stage=="before" else "postaction"
            item[key+"_png_sha256"]=sha256(png).hexdigest()
            item[key+"_rgba_sha256"]=_rgba_sha(png)
            blobs[s["case_id"]+"-"+stage+".png"]=png
        data["cases"].append(item)
    return data,blobs


def owner_environment(root):
    owner,phase=fixtures(root)
    h242=root/"dragula-native-prospects-test"
    h242.mkdir()
    data,blobs=sample_data()
    for name,blob in blobs.items():
        (h242/name).write_bytes(blob)
    (h242/"PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json").write_text(
        json.dumps(data,ensure_ascii=False),encoding="utf-8")
    h236=root/"earlier-original-qa"
    h236.mkdir()
    # H236 original en/ar contexts have IDENTICAL images by design.
    prior={
        "status":"H236_ORIGINAL_DRAGULA_NATIVE_QA_NOT_ADMISSION",
        "source_repository":REPO,"source_git_sha":COMMIT,
        "source_sha256":dict(PINNED),
        "official_train_dev_admitted":False,"new_ood_approved":False,
        "model_calls":0,"optimizer_updates":0,
        "contexts":[{
            "original_native_dom_drop_verified":True,"network_attempts":0,
            "before_png_sha256":"a"*64,"before_rgba_sha256":"b"*64,
            "after_png_sha256":"c"*64,"after_rgba_sha256":"d"*64,
        } for _ in range(2)],
    }
    (h236/"receipt.json").write_text(json.dumps(prior),encoding="utf-8")
    return owner,phase,h242,h236,data


class H244PrivateOverlapSyntheticTests(unittest.TestCase):
    def test_realistic_sha_only_synthetic_audit_is_not_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            owner,phase,h242,h236,_data=owner_environment(root)
            result=audit_owner_research_overlap(owner,h242,[root],h236)
            self.assertEqual(result["source_native_private_cases_checked"],8)
            self.assertEqual(result["new_private_png_frames_compared"],16)
            self.assertEqual(result["new_private_decoded_rgba_frames_compared"],16)
            self.assertEqual(result["new_private_media_bytes_verified"],16)
            self.assertEqual(result["other_quarantined_case_sources"],20)
            self.assertEqual(result["earlier_h236_native_research_frames"],2)
            self.assertEqual(result["pre_h173_known_png_exclusions"],1)
            for key in ("training_approved","dev_approved","ood_approved",
                        "historical_completeness_proven","old_private_media_or_gold_opened",
                        "h173_consumed_ood_opened"):
                self.assertFalse(result[key])

    def test_h236_duplicate_bilingual_visuals_are_de_duplicate_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _owner,_phase,_h242,h236,_=owner_environment(root)
            png,rgba=historical_h236_research(h236)
            self.assertEqual(len(png),2)
            self.assertEqual(len(rgba),2)

    def test_new_h242_collides_with_earlier_research_cohort_png(self):
        data,_=sample_data()
        same_png=data["cases"][0]["preaction_png_sha256"]
        with self.assertRaisesRegex(ValueError,"duplicates known"):
            verify_h242_receipt(data,{same_png},set())

    def test_new_h242_collides_with_earlier_research_cohort_rgba(self):
        data,_=sample_data()
        same_rgba=data["cases"][4]["postaction_rgba_sha256"]
        with self.assertRaisesRegex(ValueError,"duplicates known"):
            verify_h242_receipt(data,set(),{same_rgba})

    def test_duplicated_new_frame_or_modified_task_fails_closed(self):
        for mutation in ("duplicate_png","duplicate_rgba","reference_task","locale"):
            data,_=sample_data()
            if mutation=="duplicate_png":
                data["cases"][7]["postaction_png_sha256"]=data["cases"][0]["preaction_png_sha256"]
            elif mutation=="duplicate_rgba":
                data["cases"][7]["postaction_rgba_sha256"]=data["cases"][0]["preaction_rgba_sha256"]
            elif mutation=="reference_task":
                data["cases"][0]["instruction"]="Do another unregistered action"
            else:
                data["cases"][0]["language"]="ar"
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                verify_h242_receipt(data,set(),set())

    def test_any_source_or_split_promotion_denied(self):
        for key in ("official_train_dev_dataset_admitted",
                    "train_eligible","dev_eligible","ood_eligible",
                    "train_optimizer_approved",
                    "full_historical_source_and_visual_history_proven",
                    "transitive_license_and_media_rights_audited"):
            data,_=sample_data()
            data[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_h242_receipt(data,set(),set())

    def test_corrupted_new_private_png_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            owner,phase,h242,h236,data=owner_environment(root)
            filename=data["cases"][0]["case_id"]+"-before.png"
            (h242/filename).write_bytes(b"Corrupted newly created private PNG")
            with self.assertRaisesRegex(ValueError,"image bytes changed"):
                audit_owner_research_overlap(owner,h242,[root],h236)

    def test_malformed_or_duplicated_source_metadata_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            owner,phase,h242,h236,data=owner_environment(root)
            p=h242/"PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json"
            content=p.read_text(encoding="utf-8")
            # Strict parser must reject ambiguous duplicated JSON keys.
            p.write_text(content.replace(
                '"train_eligible": false',
                '"train_eligible": true,"train_eligible": false',1),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_owner_research_overlap(owner,h242,[root],h236)

    def test_reject_h173_as_old_phase_without_opening_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            owner,phase,h242,h236,_=owner_environment(root)
            (root/"Hala-H173-Private").mkdir()
            # Only H173 present in a different parent => not a valid
            # history; explicit fail closed before opening any OOD file.
            other=root/"only-h173"
            other.mkdir()
            (other/"Hala-H173-Private").mkdir()
            with self.assertRaisesRegex(ValueError,"no eligible pre-H173"):
                audit_owner_research_overlap(owner,h242,[other],h236)


if __name__ == "__main__":
    unittest.main()
