"""CPU-only deliberately synthetic H56 receipt mutation unit controls. NO GPU."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from h56.real_drag_baseline import (MODEL_ID,MODEL_REVISION,MODEL_SCHEMA,
    PROTOCOL,DEV_SOURCE_SHA,ATTEMPTS_SCHEMA,source_inputs)
from h56.verify_colab_receipt import verify

SOURCE=Path(__file__).resolve().parents[1]/"h55/assets"
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(o):return (json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf8")

class FakeReceiptConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.scenes=source_inputs(SOURCE)
        self.output=[]
        for scene in self.scenes:
            raw='UNIT_TEST_NO_MODEL_PREDICTION'
            self.output.append(dict(id=scene["id"],language=scene["language"],
                image_sha256=scene["image_sha256"],input_prompt_sha256=sha(scene["prompt"].encode("utf8")),
                output_raw=raw,output_sha256=sha(raw.encode("utf8")),elapsed_seconds=0.01,
                output_tokens=2,status="real_neural_raw"))
        self.raw=dict(schema=ATTEMPTS_SCHEMA,source_sha256=DEV_SOURCE_SHA,
            origin="provider_raw_model_outputs",
            cases=[dict(id=v["id"],image_sha256=v["image_sha256"],raw_output=v["output_raw"])
                   for v in self.output])
        self.run=dict(schema=MODEL_SCHEMA,model_id=MODEL_ID,revision=MODEL_REVISION,
            evidence="real_inference",source_sha256=DEV_SOURCE_SHA,
            coordinate_protocol=PROTOCOL,split="public_dev",blind_test=False,
            model_training_executed=False,adapters_loaded=False,
            provider="interactive_approved_free_colab",gpu="FAKE_CPU_ONLY_UNIT_TEST",
            vram_bytes=16000000000,peak_vram_bytes=1234,
            package_versions={"torch":"unit-mock","transformers":"5.16.1"},
            real_generated_cases=self.output,max_cases=24,source_cases=24,
            status="completed",independent_model_performance_qualified=False)
        self.save()

    def save(self):
        self.rawfile=self.folder/"raw-attempts.json"
        self.runfile=self.folder/"run-result.json"
        self.rawfile.write_bytes(canon(self.raw))
        self.run["raw_attempts_sha256"]=sha(self.rawfile.read_bytes())
        self.runfile.write_bytes(canon(self.run))

    def test_consistency_is_not_independent_remote_gpu_attestation(self):
        result=verify(SOURCE,self.runfile,self.rawfile)
        self.assertEqual(result["source_bound_real_raw_outputs"],24)
        self.assertFalse(result["actual_remote_gpu_independently_attested"])
        self.assertFalse(result["edge_task_success_verified"])
        self.assertFalse(result["blind_test_qualified"])
        self.assertFalse(result["model_training_executed"])

    def test_tampered_raw_text_or_hash_fails(self):
        self.raw["cases"][0]["raw_output"]="FORGED_RESULT"
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)

    def test_mismatched_revision_and_runner_provenance_fails(self):
        self.run["revision"]="0"*40
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)

    def test_fake_partial_and_wrong_source_fail(self):
        self.run["status"]="diagnostic_partial_not_baseline"
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)
        self.run["status"]="completed"
        self.run["source_sha256"]="0"*64
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)

    def test_duplicate_or_swapped_case_fail(self):
        self.raw["cases"][0]=copy.deepcopy(self.raw["cases"][1])
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)

    def test_nonfinite_timing_and_token_count_fail(self):
        self.run["real_generated_cases"][0]["elapsed_seconds"]=float("nan")
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)
        self.run["real_generated_cases"][0]["elapsed_seconds"]=0.01
        self.run["real_generated_cases"][0]["output_tokens"]=193
        self.save()
        with self.assertRaises(ValueError):verify(SOURCE,self.runfile,self.rawfile)

if __name__=="__main__":
    unittest.main()
