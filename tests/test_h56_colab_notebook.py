"""Static, source-pinned reproducibility tests for REAL Colab run instructions."""
import hashlib,json,unittest
from pathlib import Path
from h56.real_drag_baseline import MODEL_ID,MODEL_REVISION,DEV_SOURCE_SHA
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"h56/colab-h56-actual-qwen-drag-baseline.ipynb"
RUNNER=ROOT/"h56/real_drag_baseline.py"

class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads(FILE.read_text("utf8"))
        cls.code=["".join(c["source"]) for c in cls.nb["cells"] if c["cell_type"]=="code"]
        cls.md="\n".join("".join(c["source"]) for c in cls.nb["cells"] if c["cell_type"]=="markdown")
    def test_valid_nbformat_4_six_cells_and_unexecuted(self):
        self.assertEqual(self.nb["nbformat"],4)
        self.assertEqual(len(self.nb["cells"]),6)
        self.assertEqual(len(self.code),5)
        self.assertTrue(all(c["outputs"]==[] and c["execution_count"] is None
               for c in self.nb["cells"] if c["cell_type"]=="code"))
        self.assertEqual(self.nb["metadata"]["kernelspec"]["name"],"python3")

    def test_all_python_cells_compile_cpu_only(self):
        for i,source in enumerate(self.code):
            with self.subTest(cell=i):
                compile(source,f"colab-h56-cell-{i}", "exec")

    def test_exact_full_runner_source_digest(self):
        runner_sha=hashlib.sha256(RUNNER.read_bytes()).hexdigest()
        self.assertIn(runner_sha,self.code[2])
        self.assertIn("h56/real_drag_baseline.py",self.code[2])
        self.assertIn("source_inputs('h55/assets')",self.code[2])

    def test_real_model_running_is_not_fake_completion(self):
        self.assertIn("Qwen3-VL-2B",self.md)
        self.assertIn(MODEL_REVISION,self.md)
        self.assertIn("run('h55/assets',out,max_cases=24)",self.code[3])
        self.assertIn("assert len(report['real_generated_cases'])==24",self.code[3])
        self.assertIn("files.download(archive)",self.code[4])
        self.assertNotIn("synthetic_unit_fixture",self.code[3])
        self.assertNotIn("capture.json",self.code[3])

    def test_free_gpu_and_no_paid_jobs_enforced(self):
        self.assertIn("'COLAB_RELEASE_TAG' in os.environ",self.code[0])
        self.assertIn("torch.cuda.is_available()",self.code[0])
        self.assertIn("subprocess.run(['nvidia-smi'],check=True)",self.code[0])
        self.assertIn("HALA_REMOTE_BASELINE_ALLOWED",self.code[3])
        self.assertIn("KAGGLE_KERNEL_RUN_TYPE",self.code[0])
        self.assertIn("FREE GPU",self.md)
        self.assertNotIn("hf jobs",self.code[0].lower())
        self.assertEqual(DEV_SOURCE_SHA,"707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6")

if __name__=="__main__":
    unittest.main()
