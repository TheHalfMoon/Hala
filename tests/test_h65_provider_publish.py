"""H65 private provider publication is simulated, never mutates Kaggle."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h65.provider_publish import publish_latest
from training.checkpoints import publish_checkpoint, load_checkpoint


TOKEN = "private-test-token-not-for-storage"


def fixture(root):
    state = {
        "config_sha256": "1" * 64, "dataset_sha256": "2" * 64,
        "manifest_sha256": "3" * 64, "step": 2, "max_steps": 5,
        "metrics": [{"step": i, "loss": 1.0, "seconds": 1.0} for i in (1, 2)],
        "changed_parameter_tensors": 1,
        "peak_vram_bytes": 1024,
        "runtime": {},
    }
    def writer(path):
        (path / "adapter").mkdir()
        (path / "adapter" / "adapter_model.safetensors").write_bytes(b"test")
        (path / "resume.pt").write_bytes(b"optimizer")
    publish_checkpoint(root, state, writer)


class PrivateReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "checkpoint"
        self.work = Path(self.temp.name) / "report"
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def approved():
        return {"provider_published": True, "provider_is_private": True,
                "provider_readback_verified": True,
                "anonymous_provider_http_status": 403}

    def test_verified_publication_keeps_host_copy_by_default(self):
        uploader = mock.Mock(return_value=self.approved())
        result = publish_latest(self.root, TOKEN, self.work, uploader=uploader)
        self.assertTrue(result["provider_readback_verified"])
        self.assertFalse(result["host_copy_removed"])
        self.assertTrue(result["private_provider_ref"].startswith("abdulazizshehri/hala-h65-"))
        self.assertIn("PRIVATE", uploader.call_args.args[2])
        load_checkpoint(self.root, None)

    def test_explicit_cleanup_only_after_private_readback(self):
        uploader = mock.Mock(return_value=self.approved())
        result = publish_latest(self.root, TOKEN, self.work, uploader=uploader,
                                remove_host_copy=True)
        self.assertTrue(result["host_copy_removed"])
        self.assertFalse((self.root / "latest.json").exists())
        self.assertTrue((self.work / "private-readback-receipt.json").exists())

    def test_public_provider_or_anonymous_access_never_deletes(self):
        for detail in (
            {"provider_is_private": False},
            {"anonymous_provider_http_status": 200},
            {"provider_readback_verified": False},
        ):
            with self.subTest(detail=detail):
                reply = {**self.approved(), **detail}
                with self.assertRaisesRegex(ValueError, "NOT verified"):
                    publish_latest(self.root, TOKEN, self.work,
                                   uploader=lambda *args: reply,
                                   remove_host_copy=True)
                load_checkpoint(self.root, None)

    def test_missing_token_refuses_upload(self):
        uploader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "token required"):
            publish_latest(self.root, "", self.work, uploader=uploader)
        uploader.assert_not_called()

    def test_receipt_never_contains_secret_token(self):
        publish_latest(self.root, TOKEN, self.work,
                       uploader=lambda *args: self.approved())
        content = (self.work / "private-readback-receipt.json").read_text()
        self.assertNotIn(TOKEN, content)
        self.assertEqual(json.loads(content)["step"], 2)


if __name__ == "__main__":
    unittest.main()
