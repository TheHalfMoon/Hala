"""H74 concurrent bootstrap accepts only two independently completed SHA proofs."""
import unittest
from unittest import mock

from h74 import concurrent_bootstrap as h
from h74.remote_prefetch import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA

ZIP="f"*64
GIT="a"*40


def source(*args,**kwargs):
    return {"status":"REMOTE_ZIP_SHA_VERIFIED","zip_sha256":ZIP,"git_sha":GIT,
            "gpu_training_executed":False,"uploaded_chunks":2,"resumed_verified_chunks":0}


def model(*args,mode="full",**kwargs):
    return {"status":"H74_METADATA_VERIFIED" if mode=="metadata" else "H74_MODEL_SHA_VERIFIED",
            "model_id":MODEL_ID,"revision":MODEL_REVISION,
            "model_weights_verified":mode=="full",
            "model_weights_sha256":WEIGHTS_SHA if mode=="full" else None,
            "gpu_training_executed":False}


class ConcurrentQualificationTests(unittest.TestCase):
    def test_full_requires_independent_model_and_stage_proofs(self):
        with mock.patch.object(h,"MODEL_REVISION",MODEL_REVISION):
            result=h.bootstrap("source.bin",ZIP,GIT,"existing-t4",
                               uploader=source,model_prefetch=model,session_check=lambda *a,**k:"T4")
        self.assertEqual(result["status"],"MODEL_AND_TRAIN_ZIP_SHA_VERIFIED")
        self.assertEqual(result["weights_sha256"],WEIGHTS_SHA)
        self.assertFalse(result["checkpoint_published"])
        self.assertEqual(result["gpu_optimizer_steps"],0)

    def test_cpu_smoke_cannot_claim_weights_or_gpu_training(self):
        result=h.bootstrap("tiny.bin",ZIP,GIT,"cpu-research",cpu_smoke=True,
                           uploader=source,model_prefetch=model,session_check=lambda *a,**k:"CPU")
        self.assertEqual(result["status"],"METADATA_TRANSPORT_SMOKE_VERIFIED")
        self.assertIsNone(result["weights_sha256"])
        self.assertFalse(result["remote_h67_rederivation_executed"])

    def test_unverified_model_and_corrupt_training_zip_fail(self):
        def wrong_model(session,mode="full"):
            r=model(session,mode=mode)
            r["model_weights_sha256"]="0"*64
            return r
        with self.assertRaisesRegex(ValueError,"metadata or full"):
            h.prefetch("any-session",mode="other")
        for broken in (wrong_model,):
            def checked_prefetch(*args,**kwargs):
                value=broken(*args,**kwargs)
                if value["model_weights_sha256"]!=WEIGHTS_SHA:
                    raise ValueError("Colab did not SHA-verify complete Qwen model weights")
                return value
            with self.assertRaisesRegex(ValueError,"SHA-verify"):
                h.bootstrap("file",ZIP,GIT,"t4",session_check=lambda *a,**k:"T4",
                            uploader=source,model_prefetch=checked_prefetch)
        def corrupt(*args,**kwargs):
            q=source(*args,**kwargs)
            q["zip_sha256"]="0"*64
            return q
        with self.assertRaisesRegex(ValueError,"transport proof"):
            h.bootstrap("file",ZIP,GIT,"t4",session_check=lambda *a,**k:"T4",
                        uploader=corrupt,model_prefetch=model)

    def test_worker_bounds_refused_before_any_provider_action(self):
        network=mock.Mock(side_effect=AssertionError("provider must not run"))
        for workers in (0,5,True,1.5):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError,"bounded"):
                    h.bootstrap("source",ZIP,GIT,"session",workers=workers,
                                session_check=network)
        network.assert_not_called()


if __name__=="__main__":
    unittest.main()
