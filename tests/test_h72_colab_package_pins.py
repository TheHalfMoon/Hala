"""H72 regression: Colab pip pinning must not compare stale sys.modules."""
import importlib.metadata
import subprocess
import unittest
from unittest import mock

from h62 import remote_h62_launch as launch


class PinnedPackageInspection(unittest.TestCase):
    def test_versions_use_installed_distributions_not_cached_module_objects(self):
        known={"torch":"2.11.0+cu130", "transformers":"5.20.2",
               "peft":"0.21.2", "Pillow":"11.3.0"}
        with mock.patch.object(launch.metadata,"version",side_effect=lambda name: known[name]):
            observed=launch.package_versions()
        self.assertEqual(observed["transformers"],"5.20.2")
        self.assertEqual(observed["PIL"],"11.3.0")
        self.assertEqual(observed["torch"],"2.11.0+cu130")

    def test_fresh_pip_pin_admitted_and_rechecked_in_fresh_python(self):
        installed={"torch":"2.11.0+cu130", "transformers":"5.20.2",
                   "peft":"0.21.2", "Pillow":"11.3.0"}
        commands=[]
        def fake_process(args,**kwargs):
            commands.append(args)
            if "pip" in args and "install" in args:
                self.assertIn("--no-deps",args)
                self.assertIn("transformers==5.16.1",args)
                self.assertIn("peft==0.20.0",args)
                installed["transformers"]="5.16.1"
                installed["peft"]="0.20.0"
            elif "-c" in args:
                self.assertIn("Qwen3VLForConditionalGeneration",args[-1])
            else:
                raise AssertionError("unapproved package process")
            return subprocess.CompletedProcess(args,0,"","")
        with mock.patch.object(launch.metadata,"version",side_effect=lambda name: installed[name]),mock.patch.object(launch.subprocess,"run",side_effect=fake_process):
            result=launch.ensure_packages()
        self.assertEqual(result["transformers"],"5.16.1")
        self.assertEqual(result["peft"],"0.20.0")
        self.assertEqual(len(commands),2)
        self.assertIn("-c",commands[-1])

    def test_failed_fresh_import_never_counts_as_qualified(self):
        installed={"torch":"2.11.0+cu130", "transformers":"5.16.1",
                   "peft":"0.20.0", "Pillow":"11.3.0"}
        with mock.patch.object(launch.metadata,"version",side_effect=lambda name: installed[name]),mock.patch.object(launch.subprocess,"run",return_value=subprocess.CompletedProcess(["python","-c"],1,"","ImportError")):
            with self.assertRaisesRegex(RuntimeError,"fresh Python"):
                launch.ensure_packages()


if __name__=="__main__":
    unittest.main()
