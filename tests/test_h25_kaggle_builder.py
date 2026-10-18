"""H25 prospective private Kaggle notebook generation: CPU-only safeguards."""
import json
from pathlib import Path
import tempfile
import unittest

from tools import build_h25_kaggle as notebook

SHA = "a"*40


class H25KaggleNotebookTests(unittest.TestCase):
    def test_requires_exact_reviewed_full_commit(self):
        with tempfile.TemporaryDirectory() as parent:
            for bad in ("main","ps/h25-baseline-admission","abcdef","A"*40):
                with self.subTest(bad=bad):
                    path=Path(parent)/("bad-"+str(len(bad)))
                    with self.assertRaises(ValueError):
                        notebook.generate(bad,path)
            self.assertFalse(any(Path(parent).iterdir()))

    def test_private_gpu_and_protocol_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"candidate"
            created=notebook.generate(SHA,output)
            metadata=json.loads((output/"kernel-metadata.json").read_text(encoding="utf-8"))
            nb=json.loads((output/"hala-h25-mind-dev.ipynb").read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"] and metadata["enable_internet"])
            self.assertFalse(metadata["enable_tpu"])
            self.assertEqual(metadata["model_sources"],[])
            self.assertEqual(metadata["dataset_sources"],[])
            self.assertEqual(nb["metadata"]["hala_source_commit"],SHA)
            self.assertEqual(nb["metadata"]["hala_split"],"dev")
            self.assertEqual(nb["metadata"]["hala_protocol"],"h25-seven-pair-dev-first-v1")
            self.assertFalse(created["submitted"] or created["inference_run"])
            self.assertFalse(nb["metadata"]["hala_gpu_submitted"])

    def test_exact_head_at_remote_before_any_model_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"notebook"
            notebook.generate(SHA,root)
            obj=json.loads((root/"hala-h25-mind-dev.ipynb").read_text(encoding="utf-8"))
            self.assertEqual(len(obj["cells"]),5)
            code=["".join(c["source"]) for c in obj["cells"] if c["cell_type"]=="code"]
            self.assertIn(SHA,code[2])
            self.assertIn("git","".join(code))
            self.assertIn("rev-parse",code[2])
            self.assertIn("--source-commit",code[3])
            self.assertIn("h25.baseline",code[3])
            self.assertNotIn("--split test","".join(code).lower())
            self.assertIn("HALA_REMOTE_DEVELOPMENT_ALLOWED",code[2])
            self.assertIn("KAGGLE_KERNEL_RUN_TYPE",code[0])
            # The pip magic is notebook-only syntax, so compile other code cells.
            for i,src in enumerate(code):
                if i!=1:
                    compile(src,"h25_notebook_cell_"+str(i),"exec")

    def test_notebook_never_overwrites_saved_version(self):
        with tempfile.TemporaryDirectory() as parent:
            path=Path(parent)/"saved"
            notebook.generate(SHA,path)
            original=(path/"hala-h25-mind-dev.ipynb").read_bytes()
            with self.assertRaises(FileExistsError):
                notebook.generate(SHA,path)
            self.assertEqual((path/"hala-h25-mind-dev.ipynb").read_bytes(),original)

    def test_documented_module_command_actually_executes(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"module-entrypoint"
            repo=Path(__file__).resolve().parents[1]
            proc=subprocess.run(
                [sys.executable,"-m","tools.build_h25_kaggle",
                 "--source-commit",SHA,"--output",str(output)],
                cwd=repo,capture_output=True,text=True,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertTrue((output/"hala-h25-mind-dev.ipynb").is_file())
            result=json.loads(proc.stdout.strip())
            self.assertFalse(result["submitted"])
            self.assertFalse(result["inference_run"])

    def test_no_raw_oracle_export_in_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"n"
            notebook.generate(SHA,path)
            contents=(path/"hala-h25-mind-dev.ipynb").read_text(encoding="utf-8")
            for forbidden in ('"oracle_private"', '"expected"', '"split": "test"',
                              "delete_everything","H24 TEST"):
                self.assertNotIn(forbidden,contents)


if __name__=="__main__":
    unittest.main()
