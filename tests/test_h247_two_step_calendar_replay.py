"""H247 action-horizon correctness from synthetic receipts, no private media."""
from copy import deepcopy
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS,parse_actions
from h246.original_calendar_native import CASES,COMMIT,FILES,REPO
from h247.two_step_calendar_replay import (
    STAGES,STATUS,one_click_action,validate_two_step_replay,
    verify_original_h246_research,
)


def old_research():
    return {
        "status":"H246_THIRD_ORIGINAL_CALENDAR_NATIVE_RESEARCH_NOT_ADMITTED",
        "source_repository":REPO,"source_git_sha":COMMIT,
        "original_source_sha256":dict(FILES),
        "original_mit_javascript_unmodified":True,
        "self_authored_html_fixture":True,
        "historical_source_visual_history_complete":False,
        "transitive_assets_rights_complete":False,
        "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
        "model_calls":0,"optimizer_steps":0,"cases":[{
            "case_id":x["id"],"language":x["lang"],
            "instruction":x["instruction"],"calendar_day":x["day"],
            "real_native_calendar_selection_verified":True,
            "network_requests":0,
            "calendar_click_point":[110,220],
            "confirmed_input_value":f"2026-10-{x['day']:02}",
            "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
            "preaction_png_sha256":f"{i+1:064x}",
            "preaction_rgba_sha256":f"{i+20:064x}",
            "postaction_png_sha256":f"{i+100:064x}",
            "postaction_rgba_sha256":f"{i+120:064x}",
        } for i,x in enumerate(CASES)],
    }


def replay_case(spec):
    first=(f"Select day {spec['day']} from the open calendar."
           if spec["lang"]=="en" else
           f"اختر اليوم {spec['day']} من التقويم المفتوح.")
    second=("Confirm the chosen date with the OK button."
            if spec["lang"]=="en" else
            "أكد التاريخ المحدد بالضغط على زر OK.")
    return {
        "case_id":spec["id"],"language":spec["lang"],
        "source_repository":REPO,"selected_day":spec["day"],
        "confirmed_date":f"2026-10-{spec['day']:02}",
        "original_native_calendar_verified":True,
        "network_requests":0,
        "frames":{key:{"png_sha256":f"{i+1:064x}",
                       "rgba_sha256":f"{i+30:064x}"}
                  for i,key in enumerate(STAGES)},
        "steps":[{
            "step_index":i+1,"instruction":instruction,
            "click_point":point,"native_action":one_click_action(point),
            "source_frame":STAGES[i],"result_frame":STAGES[i+1],
            "research_only":True,
            "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
        } for i,(instruction,point) in enumerate([
            (first,[300,260]),(second,[580,640])])],
    }


class H247TwoStepCalendarActionHorizonTests(unittest.TestCase):
    def test_h199_requires_exactly_one_action_per_input(self):
        for bad in (
            '{"actions":[{"kind":"click","point":[300,260]},'
            '{"kind":"click","point":[580,640]}]}',
            '{"actions":[]}',
        ):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                parse_actions(bad,training_target=True)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS,frozenset())

    def test_each_step_is_an_independent_h199_click_action(self):
        for case in CASES:
            with self.subTest(source_case=case["id"]):
                x=replay_case(case)
                validate_two_step_replay(x,case)
                self.assertEqual(len(x["steps"]),2)
                self.assertNotEqual(x["steps"][0]["click_point"],
                                    x["steps"][1]["click_point"])
                self.assertEqual(x["steps"][0]["source_frame"],STAGES[0])
                self.assertEqual(x["steps"][1]["source_frame"],STAGES[1])

    def test_original_h246_receipt_must_remain_quarantined(self):
        record=old_research()
        verify_original_h246_research(record)
        for field in ("train_eligible","dev_eligible","ood_eligible",
                      "transitive_assets_rights_complete",
                      "historical_source_visual_history_complete"):
            data=deepcopy(record)
            data[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_original_h246_research(data)

    def test_unexpected_source_sha_genuine_oracle_or_locale_denied(self):
        original=old_research()
        for field,val in (
            ("source_git_sha","0"*40),
            ("source_repository","mock/wrong"),
            ("status","H246_OFFICIALLY_ADMITTED"),
        ):
            data=deepcopy(original);data[field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_original_h246_research(data)
        for field,val in (
            ("instruction","reference label leaked"),
            ("language","en"),
            ("calendar_day",17),
            ("real_native_calendar_selection_verified",False),
            ("train_eligible",True),
            ("confirmed_input_value","2026-10-28"),
        ):
            data=deepcopy(original);data["cases"][1][field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_original_h246_research(data)

    def test_identical_old_visual_or_forged_sha_denied(self):
        record=old_research()
        record["cases"][7]["postaction_png_sha256"] = record["cases"][0]["preaction_png_sha256"]
        with self.assertRaises(ValueError):
            verify_original_h246_research(record)
        record=old_research()
        record["cases"][0]["preaction_rgba_sha256"]="zz"*32
        with self.assertRaises(ValueError):
            verify_original_h246_research(record)

    def test_missing_intermediate_or_combined_action_denied(self):
        x=replay_case(CASES[0])
        del x["frames"][STAGES[1]]
        with self.assertRaises(ValueError):
            validate_two_step_replay(x,CASES[0])
        x=replay_case(CASES[0])
        x["steps"][1]["native_action"]={"actions":[
            {"kind":"click","point":[300,260]},
            {"kind":"click","point":[580,640]},
        ]}
        with self.assertRaises(ValueError):
            validate_two_step_replay(x,CASES[0])

    def test_step_transition_reorder_and_role_promotion_denied(self):
        for field,val in (
            ("source_frame",STAGES[0]),
            ("result_frame",STAGES[0]),
            ("step_index",1),
            ("train_eligible",True),
            ("dev_eligible",True),
            ("ood_eligible",True),
            ("research_only",False),
        ):
            x=replay_case(CASES[0])
            x["steps"][1][field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_two_step_replay(x,CASES[0])

    def test_new_intermediate_visual_must_be_distinct(self):
        x=replay_case(CASES[0])
        x["frames"][STAGES[1]]["rgba_sha256"] = x["frames"][STAGES[0]]["rgba_sha256"]
        with self.assertRaises(ValueError):
            validate_two_step_replay(x,CASES[0])

    def test_native_two_step_click_coordinates_must_be_visible(self):
        for bad in ([960,200],[-1,4],[12.5,4],[300,720],[300,300,300]):
            with self.subTest(point=bad),self.assertRaises(ValueError):
                one_click_action(bad)
        for spec in CASES:
            self.assertTrue(spec["instruction"])


if __name__=="__main__":
    unittest.main()
