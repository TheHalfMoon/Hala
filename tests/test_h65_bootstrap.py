"""H65: pinned packages cannot resolve large GPU dependencies implicitly."""
import sys
import types
import subprocess
import unittest
from unittest import mock

from h62 import remote_h62_launch as launch


def installed(transformers="5.20.1", peft="0.21.2"):
    return {"torch": "2.11.0+cu128", "transformers": transformers,
            "peft": peft, "PIL": "11.3.0"}


class BootstrapTests(unittest.TestCase):
    def imports(self):
        trans = types.ModuleType("transformers")
        for name in ("AutoProcessor", "Qwen3VLForConditionalGeneration"):
            setattr(trans, name, type(name, (), {}))
        peft = types.ModuleType("peft")
        for name in ("LoraConfig", "PeftModel"):
            setattr(peft, name, type(name, (), {}))
        return {"transformers": trans, "peft": peft}

    def test_pin_only_install_never_triggers_large_torch_dependency_resolution(self):
        with (
            mock.patch.object(launch, "package_versions",
                              side_effect=[installed(), installed("5.16.1", "0.20.0")]),
            mock.patch.object(launch.subprocess, "run") as process,
            mock.patch.dict(sys.modules, self.imports()),
        ):
            versions = launch.ensure_packages()
        self.assertEqual(versions["peft"], "0.20.0")
        self.assertEqual(process.call_count, 2)
        called = process.call_args_list[0].args[0]
        self.assertIn("--no-deps", called)
        self.assertIn("transformers==5.16.1", called)
        self.assertIn("peft==0.20.0", called)
        self.assertNotIn("torch==2.11.0", called)
        self.assertEqual(process.call_args_list[0].kwargs["timeout"], 180)
        self.assertIn("-c", process.call_args_list[1].args[0])
        self.assertEqual(process.call_args_list[1].kwargs["timeout"], 120)

    def test_no_install_if_all_pins_already_present(self):
        with (
            mock.patch.object(launch, "package_versions",
                              return_value=installed("5.16.1", "0.20.0")),
            mock.patch.object(launch.subprocess, "run") as process,
            mock.patch.dict(sys.modules, self.imports()),
        ):
            launch.ensure_packages()
        process.assert_called_once()
        self.assertIn("-c", process.call_args.args[0])
        self.assertNotIn("pip", process.call_args.args[0])

    def test_bad_post_install_pin_refuses_before_model_download(self):
        with (
            mock.patch.object(launch, "package_versions", return_value=installed()),
            mock.patch.object(launch.subprocess, "run"),
            mock.patch.dict(sys.modules, self.imports()),
        ):
            with self.assertRaisesRegex(RuntimeError, "pin failed"):
                launch.ensure_packages()

    def test_missing_transitive_import_is_a_failure_not_automatic_unbounded_pip(self):
        # A cached sys.modules value does not prove fresh interpreter
        # importability. Simulate the bounded fresh interpreter returning a
        # dependency failure without running a real subprocess or pip.
        with (
            mock.patch.object(launch, "package_versions",
                              return_value=installed("5.16.1", "0.20.0")),
            mock.patch.object(launch.subprocess, "run",
                              return_value=subprocess.CompletedProcess(
                                  ["python", "-c"], 1, "", "ImportError")) as process,
            mock.patch.dict(sys.modules, {"transformers": None, "peft": None}),
        ):
            with self.assertRaisesRegex(RuntimeError, "training imports unavailable"):
                launch.ensure_packages()
        process.assert_called_once()
        self.assertIn("-c", process.call_args.args[0])
        self.assertNotIn("pip", process.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
