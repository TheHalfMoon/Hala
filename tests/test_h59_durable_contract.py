"""MiniMax durable plan CPU-only tests; never talk to provider or run GPU."""
import copy
import json
import unittest
from pathlib import Path
from h59 import durable_contract as c

PLAN=Path(__file__).resolve().parents[1]/"h59/full_source_windows.json"

class H59DurableWorkflowControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(PLAN.read_text("utf8"))
        cls.window=cls.plan["windows"][0]

    def test_deterministic_source_key_and_initial(self):
        a=c.initial(self.window)
        self.assertEqual(a,c.initial(self.window))
        self.assertEqual(a["job_id"],c.key(self.window))
        self.assertEqual(a["state"],"PLANNED")
        self.assertFalse(a["authorized_free_compute"])
        self.assertFalse(a["model_training_qualified"])
        self.assertEqual(a["model_gpu_steps"],0)
        self.assertEqual(c.check_manifest_jobs(self.plan,[a])["registered_jobs"],1)

    def test_exact_window_verified_requires_real_sha(self):
        a=c.initial(self.window)
        bad={**a,"state":"WINDOW_SHA_VERIFIED"}
        with self.assertRaises(ValueError):c.validate_transition(a,bad,self.window)
        good={**bad,"original_window_sha256":"1"*64}
        r=c.validate_transition(a,good,self.window)
        self.assertEqual(r["status"],"WINDOW_SHA_VERIFIED")
        self.assertFalse(r["model_training_in_progress"])
        self.assertEqual(c.check_manifest_jobs(self.plan,[good])["registered_jobs"],1)

    def test_no_skipping_or_negative_counts(self):
        a=c.initial(self.window)
        candidate={**a,"state":"ORIGINAL_MEMBERS_VERIFIED",
                   "original_window_sha256":"1"*64}
        with self.assertRaises(ValueError):c.validate_transition(a,candidate,self.window)
        candidate={**a,"original_member_count":-1}
        with self.assertRaises(ValueError):c.validate_transition(a,candidate,self.window)

    def test_pair_gate_requires_original_source_member_receipt(self):
        a=c.initial(self.window)
        b={**a,"state":"WINDOW_SHA_VERIFIED","original_window_sha256":"1"*64}
        c.validate_transition(a,b,self.window)
        other={**b,"state":"ORIGINAL_MEMBERS_VERIFIED",
               "original_member_manifest_sha256":"2"*64,
               "original_member_count":3,"original_source_pair_count":0}
        with self.assertRaises(ValueError):c.validate_transition(b,other,self.window)
        other["original_source_pair_count"]=1
        self.assertEqual(c.validate_transition(b,other,self.window)["status"],
                         "ORIGINAL_MEMBERS_VERIFIED")

    def test_private_requires_all_proofs_and_locks_old_provenance(self):
        a=c.initial(self.window)
        b={**a,"state":"WINDOW_SHA_VERIFIED","original_window_sha256":"1"*64}
        c1={**b,"state":"ORIGINAL_MEMBERS_VERIFIED",
            "original_member_manifest_sha256":"2"*64,
            "original_member_count":4,"original_source_pair_count":2}
        d1={**c1,"state":"PRIVATE_KAGGLE_READBACK_VERIFIED",
            "private_kaggle_ref":"abdulazizshehri/hala-new-data-private",
            "private_provider_readback_sha256":"2"*64}
        with self.assertRaises(ValueError):c.validate_transition(c1,d1,self.window)
        d1["private_provider_is_private_verified"]=True
        d1["anonymous_provider_http403_verified"]=True
        c.validate_transition(c1,d1,self.window)
        self.assertEqual(c.check_manifest_jobs(self.plan,[d1])["registered_jobs"],1)
        counterfeit={**d1,"original_window_sha256":"9"*64}
        with self.assertRaises(ValueError):c.validate_transition(d1,counterfeit,self.window)
        modified={**d1,"private_provider_readback_sha256":"f"*64}
        with self.assertRaises(ValueError):c.validate_transition(d1,modified,self.window)

    def test_source_jobs_cannot_masquerade_as_neural_model_training(self):
        a=c.initial(self.window)
        for field,value in (("model_gpu_steps",1),("model_training_qualified",True),
                            ("authorized_free_compute",True),
                            ("independent_neural_evaluation_completed",True)):
            modified={**a,field:value}
            with self.assertRaises(ValueError):c.validate_transition(a,modified,self.window)

    def test_no_regression_or_another_unplanned_source_identity(self):
        a=c.initial(self.window)
        b={**a,"state":"WINDOW_SHA_VERIFIED","original_window_sha256":"1"*64}
        with self.assertRaises(ValueError):c.validate_transition(b,a,self.window)
        x={**b,"source_revision":"0"*40}
        with self.assertRaises(ValueError):c.validate_transition(a,x,self.window)

    def test_no_duplicate_window_key_in_durable_plan(self):
        a=c.initial(self.window)
        with self.assertRaises(ValueError):c.check_manifest_jobs(self.plan,[a,copy.deepcopy(a)])
        b={**a,"window_id":"h59-fake-9999"}
        with self.assertRaises(ValueError):c.check_manifest_jobs(self.plan,[b])

if __name__=="__main__":
    unittest.main()
