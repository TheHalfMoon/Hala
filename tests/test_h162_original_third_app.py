"""H162 real third app research OOD source/provenance checks."""
import inspect
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from h162.original_third_app import spec, state_for, prompt, APP, PIN
from h162.gpu_infer import source_cases
from h162.pack_private_inference import SOURCE_SHA,TRAINED_SHA


class H162IndependentAppTests(unittest.TestCase):
    def test_third_app_is_not_training_app(self):
        self.assertEqual(APP,"react-html5-board")
        self.assertEqual(len(PIN),40)
        self.assertEqual(spec(0)["language"],"en")
        self.assertEqual(spec(1)["language"],"ar")
        self.assertEqual(spec(0)["index"],192001)
        self.assertEqual(spec(15)["index"],192016)

    def test_original_html5_data_and_real_arabic_task(self):
        case=spec(1)
        title,data=state_for(case)
        self.assertIn(title, data[0]["cards"][0]["title"])
        self.assertIn("اسحب",prompt(case,title))
        self.assertEqual(data[1]["id"],"todo")

    def test_refuse_gold_or_unrelated_GPU_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            z=Path(tmp)/"bad.zip"
            with ZipFile(z,"w") as writer:
                writer.writestr("manifest.json","{}")
            with self.assertRaises(ValueError):
                source_cases(z)
        from h162 import pack_private_inference,original_third_app
        public=inspect.getsource(pack_private_inference.run)
        self.assertIn("gold_points_uploaded",public)
        self.assertNotIn("model.train(",public)
        self.assertIn("fully_sealed_BLIND",inspect.getsource(original_third_app.run))
        self.assertEqual(len(SOURCE_SHA),64)
        self.assertEqual(len(TRAINED_SHA),64)


if __name__=="__main__":
    unittest.main()
