"""CPU tests: replay qualification only, not evidence of trained Hala."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h88_private", HERE / "h88/kaggle_private_run.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class KaggleReplayTests(unittest.TestCase):
    def test_metadata_exactly_private_research_h71(self):
        data = json.loads((HERE / "h88/kernel-metadata.json").read_text())
        self.assertTrue(data["is_private"])
        self.assertTrue(data["enable_gpu"])
        self.assertTrue(data["enable_internet"])
        self.assertEqual(data["kernel_type"], "script")
        self.assertEqual(data["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(data["dataset_sources"], [module.DATASET_REF])
        self.assertEqual(data["code_file"], "kaggle_private_run.py")
        self.assertFalse(data["model_sources"])

    def test_requires_real_kaggle_and_cuda(self):
        for env, workspace, devices in (({}, True, 2),
                                         ({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, False, 2),
                                         ({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, True, 0)):
            with self.subTest(env=env, workspace=workspace, devices=devices):
                with self.assertRaises(ValueError):
                    module.require_private_research_environment(
                        env, is_kaggle_working=workspace, cuda_devices=devices)
        module.require_private_research_environment(
            {"KAGGLE_KERNEL_RUN_TYPE": "Batch"},
            is_kaggle_working=True, cuda_devices=2)

    def test_wrong_opt_in_rejected(self):
        with self.assertRaises(ValueError):
            module.require_private_research_environment({
                "KAGGLE_KERNEL_RUN_TYPE": "Batch", "HALA_NONCOMMERCIAL_RESEARCH": "0"},
                is_kaggle_working=True, cuda_devices=2)

    def test_bounded_failure_never_leaks_provider_secrets(self):
        value = module.safe_failure("PINNED_SOURCE", RuntimeError("secret-private-token"))
        self.assertFalse(value["model_trained_verified"])
        self.assertFalse(value["provider_private_readback_verified"])
        self.assertNotIn("secret-private-token", json.dumps(value))
        self.assertEqual(value["failure_class"], "RuntimeError")
        self.assertFalse(value["commercial_use"])

    def test_write_rejects_raw_training_text(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "receipt.json"
            module.save_attempt({"schema": module.SCHEMA, "status": "STARTED"}, path=p)
            self.assertEqual(json.loads(p.read_text())["status"], "STARTED")
            with self.assertRaises(ValueError):
                module.save_attempt({"status": "error", "prompt": "private"}, path=p)

    def test_bootstrap_removes_incompatible_optional_torchao_before_install(self):
        fake = {"torch": "2.11.0+cu128", "transformers": "5.16.1",
                "peft": "0.20.0", "Pillow": "11.3.0"}
        def version(name):
            if name in fake:
                return fake[name]
            raise module.importlib.metadata.PackageNotFoundError(name)
        with mock.patch.object(module, "checked_process") as calls, \
             mock.patch.object(module.importlib.metadata, "version", side_effect=version):
            self.assertEqual(module.install_pinned_dependencies(), fake)
            self.assertEqual(calls.call_args_list[0].args[3:],
                             ("uninstall", "-y", "torchao"))
            self.assertEqual(calls.call_args_list[1].args[3], "install")

    def test_bad_torch_version_refused(self):
        fake = {"torch": "2.10.0+cu128", "transformers": "5.16.1",
                "peft": "0.20.0", "Pillow": "11.3.0"}
        with mock.patch.object(module, "checked_process"), \
             mock.patch.object(module.importlib.metadata, "version",
                               side_effect=fake.__getitem__):
            with self.assertRaisesRegex(ValueError, "pinned"):
                module.install_pinned_dependencies()


if __name__ == "__main__":
    unittest.main()
