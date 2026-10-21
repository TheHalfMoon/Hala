"""H56 source-isolated real-model runner admission tests (no GPU/weights)."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import shutil
import tempfile
from h56 import real_drag_baseline as b

ROOT=Path(__file__).resolve().parents[1]/"h55/assets"

class H56RealBaselineTests(TestCase):
    def test_exact_public_dev_24_pngs_are_source_bound(self):
        entries=b.source_inputs(ROOT)
        self.assertEqual(len(entries),24)
        self.assertEqual(sum(x["language"]=="ar" for x in entries),12)
        self.assertEqual(sum(x["language"]=="en" for x in entries),12)
        self.assertEqual(len({x["image_sha256"] for x in entries}),24)
        self.assertEqual(len({x["family"] for x in entries}),6)
        self.assertTrue(all("1024x640" in x["prompt"] for x in entries))
        self.assertTrue(all("end_point" in x["prompt"] for x in entries))
        self.assertTrue(all("source_image_pixels" in x["prompt"] for x in entries))
        self.assertEqual(b.MODEL_ID,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(b.MODEL_REVISION,"89644892e4d85e24eaac8bacfd4f463576704203")

    def test_does_not_read_original_h55_gold_endpoints(self):
        names=[]
        original=Path.read_text
        def spy(path,*args,**kwargs):
            if path.name=="capture.json":
                raise AssertionError("No DEV oracle capture is allowed in model input")
            names.append(path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,"read_text",spy):
            data=b.source_inputs(ROOT)
        self.assertEqual(len(data),24)
        self.assertNotIn("capture.json",names)

    def test_input_is_public_dev_not_training_or_blind(self):
        data=b.source_inputs(ROOT)
        self.assertTrue(all("/png/" in x["image_path"].replace("\\","/") for x in data))
        self.assertTrue(all(x["id"].startswith("h55-") for x in data))
        self.assertEqual(b.DEV_SOURCE_SHA,"707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6")

    def test_modified_plan_rejected_at_source_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"html").mkdir()
            (root/"png").mkdir()
            for file in ("source.json","plans.json"):
                shutil.copyfile(ROOT/file,root/file)
            plan=root/"plans.json"
            plan.write_bytes(plan.read_bytes()+b" ")
            with self.assertRaises(ValueError):b.source_inputs(root)

    def test_modified_first_png_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            shutil.copytree(ROOT,root/"copy")
            first=next((root/"copy/png").glob("*.png"))
            first.write_bytes(b"tamperedPNG")
            with self.assertRaises(ValueError):b.source_inputs(root/"copy")

    def test_wont_run_weights_on_windows_or_unapproved_colab(self):
        with patch.object(b.platform,"system",return_value="Windows"):
            with self.assertRaises(ValueError):b.run(ROOT,Path("never-run"))
        with patch.object(b.platform,"system",return_value="Linux"):
            with patch.dict("os.environ",{"HALA_REMOTE_BASELINE_ALLOWED":"1","COLAB_RELEASE_TAG":""}):
                with self.assertRaises(ValueError):b.validate_remote()

    def test_output_is_create_only_and_partial_is_not_full_baseline(self):
        import inspect
        src=inspect.getsource(b.run)
        self.assertIn("output.exists()",src)
        self.assertIn('"diagnostic_partial_not_baseline"',src)
        self.assertIn('max_cases==24',src)
        self.assertIn("real_generated_cases",src)
        self.assertIn("run-partial.json",src)
        self.assertNotIn("capture.json",src)
        self.assertNotIn("torch.load",src)

    def test_official_benchmark_not_promoted(self):
        self.assertEqual(b.MODEL_SCHEMA,"hala.h56.real-unchanged-qwen-native-drag-dev.v1")
        self.assertEqual(b.PROTOCOL,"original_image_pixels_1024x640_native_drag_v1")
        self.assertEqual(b.ALLOWED_GPU_PLATFORMS,("colab",))
