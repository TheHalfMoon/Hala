"""H39 CPU-only host-supplied fresh observation consistency regressions."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import time
import unittest

from PIL import Image

from hala.contracts.validation import ContractError
from hala.inference import RequestContext
from hala.inference.fresh_snapshot_review import review_same_fresh_snapshot
from hala.inference.vision_candidate import CandidateBackend
from hala.serving import Config, Service
from tests.test_serving_v2 import FakeEngine, candidate_payload, TOKEN


class FreshSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.original = candidate_payload()
        self.original["policy_request"]["task"]["goal"] = (
            "Select the action labeled 'Save' on the page.")
        self.service = Service(Config(TOKEN, "tenant-1"),
                               candidate_backend=CandidateBackend(
                                   self.engine, require_explicit_label=True))
        self.proposal = self.service.infer(
            "v2_action", self.original, RequestContext(time.monotonic()+3))["result"]
        self.current = copy.deepcopy(self.original)
        self.trusted = self.original["policy_request"]["task"]["goal"]

    def check(self):
        return review_same_fresh_snapshot(self.original,self.proposal,
                                          self.current,trusted_goal=self.trusted)

    def test_full_valid_freshly_recaptured_view_is_only_consistent_not_authorized(self):
        self.assertIsNone(self.check())
        self.assertFalse(self.proposal["execution_authorized"])
        self.assertFalse(self.proposal["provenance"]["qualified"])
        self.assertEqual(self.engine.calls,1)

    def test_changed_host_window_and_observation_context_fail_closed(self):
        changes = [
            ("window_ref",("observation","window","window_ref"),"other-window"),
            ("title",("observation","window","title"),"New modal dialog"),
            ("process",("observation","window","process"),"unrelated.exe"),
            ("protected_window",("observation","window","protected"),True),
            ("target_name",("observation","accessibility",0,"name"),"Delete"),
            ("target_ref",("observation","accessibility",0,"element_ref"),"save-other"),
            ("target_bounds",("observation","accessibility",0,"bounds"),[40,25,80,30]),
            ("target_protected",("observation","accessibility",0,"protected"),True),
            ("redaction",("observation","image","redactions"),[[20,30,10,10]]),
            ("truncated",("observation","truncation","elements_omitted"),1),
            ("coordinate_space",("observation","image","coordinate_space"),"viewport"),
            ("locale",("device","locale"),"ar-SA"),
            ("session",("device","session_id"),"session-new"),
            ("device",("device","device_id"),"device-new"),
            ("generation",("observation","generation"),"obs-2"),
            ("task_id",("task","task_id"),"task-new"),
            ("step_index",("task","step_index"),1),
            ("cancelled",("cancelled",),True),
            ("user_goal",("task","goal"),"Select the action labeled 'Delete' on the page."),
            ("actions",("allowed_actions",),["abstain"]),
            ("ceiling",("method_ceiling",),"STRUCTURED_ONLY"),
        ]
        for name,path,value in changes:
            with self.subTest(name=name):
                self.current=copy.deepcopy(self.original)
                obj=self.current["policy_request"]
                for key in path[:-1]:
                    obj=obj[key]
                obj[path[-1]]=value
                if "frame_id" == name:
                    self.current["media"]["frame_id"]=value
                with self.assertRaises((ContractError,ValueError)):
                    self.check()
        self.assertEqual(self.engine.calls,1)

    def test_self_consistently_recaptured_new_generation_is_still_stale(self):
        # Independent fresh capture is valid on its own, but does not match
        # the model's older observation/proposal. No malformed-media shortcut.
        from hala.inference.vision_candidate import parse_request
        self.current["policy_request"]["observation"]["generation"]="obs-fresh"
        self.current["media"]["frame_id"]="obs-fresh"
        request, image, eligible, _ = parse_request(self.current,"tenant-1")
        self.assertEqual(request.observation.generation,"obs-fresh")
        self.assertEqual(image.size,(800,600))
        self.assertTrue(eligible)
        with self.assertRaisesRegex(ContractError,"changed"):
            self.check()

    def test_self_consistently_recaptured_new_session_is_still_stale(self):
        from hala.inference.vision_candidate import parse_request
        self.current["policy_request"]["device"]["session_id"]="session-fresh"
        self.current["media"]["session_id"]="session-fresh"
        current, *_ = parse_request(self.current,"tenant-1")
        self.assertEqual(current.device.session_id,"session-fresh")
        with self.assertRaisesRegex(ContractError,"changed"):
            self.check()

    def test_unchanged_screenshot_with_different_accessibility_is_stale(self):
        # The screenshot digest alone does not bind observed label semantics.
        from hala.inference.vision_candidate import parse_request
        self.current["policy_request"]["observation"]["accessibility"][0]["name"]="Remove"
        parsed, *_=parse_request(self.current,"tenant-1")
        self.assertEqual(parsed.observation.accessibility[0].name,"Remove")
        self.assertEqual(self.current["media"]["sha256"],self.original["media"]["sha256"])
        with self.assertRaisesRegex(ContractError,"changed"):
            self.check()

    def test_new_screenshot_bytes_even_with_unchanged_goal_rejected(self):
        buf=io.BytesIO()
        Image.new("RGB",(800,600),"#deface").save(buf,format="PNG")
        png=buf.getvalue()
        self.current["media"]["png_base64"]=base64.b64encode(png).decode("ascii")
        self.current["media"]["sha256"]=hashlib.sha256(png).hexdigest()
        with self.assertRaises((ContractError,ValueError)):
            self.check()

    def test_independently_trusted_goal_disagreement_rejected(self):
        self.trusted="Select the action labeled 'Delete' on the page."
        with self.assertRaises((ContractError,ValueError)):
            self.check()
        self.trusted=None
        with self.assertRaises((ContractError,ValueError)):
            self.check()

    def test_candidate_tamper_authority_identity_and_unexpected_fields(self):
        for field,value in (
            ("execution_authorized",True),("verification_required",False),
            ("confidence",0.99),("status","done"),
            ("request_id","other-request"),("tenant_id","other-tenant"),
            ("task_id","other-task"),("device_id","other-device"),
            ("session_id","other-session"),("frame_id","new-frame"),
            ("media_sha256","0"*64),("actions",[]),
            ("extra","bypass")
        ):
            with self.subTest(field=field):
                self.proposal[field]=value
                with self.assertRaises((ContractError,ValueError,TypeError)):
                    self.check()
                self.proposal=self.service.infer(
                    "v2_action",self.original,RequestContext(time.monotonic()+3))["result"]

    def test_action_tamper_to_other_target_and_outside_geometry_rejected(self):
        original=copy.deepcopy(self.proposal)
        for key,value in (
            ("element_ref_hint","other"),
            ("point",[780,440]),
            ("point",[50.0,35.0]),
            ("coordinate_space","viewport"),
            ("image_scale",1.2),
            ("secret","irrelevant")
        ):
            with self.subTest(key=key,value=value):
                self.proposal=copy.deepcopy(original)
                self.proposal["actions"][0][key]=value
                with self.assertRaises((ContractError,ValueError)):
                    self.check()

    def test_actual_wrong_unique_target_remains_rejected(self):
        self.original["policy_request"]["observation"]["accessibility"].append({
            "element_ref":"cancel-1","role":"Button","name":"Cancel",
            "bounds":[150,50,100,30]})
        self.current=copy.deepcopy(self.original)
        self.engine.output='{"point":[250,108]}'
        default=Service(Config(TOKEN,"tenant-1"),
                        candidate_backend=CandidateBackend(self.engine))
        wrong=default.infer("v2_action",self.original,
                            RequestContext(time.monotonic()+3))["result"]
        self.proposal=wrong
        self.assertEqual(wrong["actions"][0]["element_ref_hint"],"cancel-1")
        with self.assertRaises((ContractError,ValueError)):
            self.check()


if __name__=="__main__":
    unittest.main()
