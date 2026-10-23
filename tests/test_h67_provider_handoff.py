"""H67 provider and Colab upload tests: all third-party calls mocked."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h67.private_stage import publish_private
from h67.upload_stage import upload
from tests.test_h67_cpu_stage import fixture
from h67.stage import make_stage

TOKEN = "fake-provider-test-token-long-enough"
GIT = "a"*40


class ProviderStub:
    OWNER = "abdulazizshehri"
    def __init__(self, *, weak=False, public=False):
        self.weak = weak
        self.public = public
        self.calls = []
    def create_private_dataset(self, work, ref, title, subtitle):
        self.calls.append(("create", ref))
        self.file_count = len(list(Path(work).iterdir()))
    def wait_until_visible(self, ref, token):
        self.calls.append(("visible", ref))
    def verify_private(self, ref, token):
        self.calls.append(("privacy", ref))
        return {"provider_is_private": not self.public,
                "anonymous_view_http_status": 200 if self.public else 403}
    def verify_readback(self, ref, token, sizes, shas, work):
        self.calls.append(("readback", ref))
        if self.weak:
            return {"provider_inventory_complete": True,
                    "provider_readback_members_byte_verified_count": 1,
                    "provider_readback_members_size_verified_count": 1}
        return {"provider_readback_files": len(sizes),
                "provider_readback_bytes_mismatched": 0}


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        work = self.root / "cpu"
        fixture(work)
        self.bundle = self.root / "stage.zip"
        self.result = make_stage(work, self.bundle, GIT, rederive=False)
    def tearDown(self):
        self.t.cleanup()
    def call(self, stub, *, delete=True, token=TOKEN):
        return publish_private(self.bundle, self.result["sha256"], GIT, token,
                               self.root / "receipts", provider=stub,
                               delete_host_copy=delete)

    def test_authenticated_private_exact_readback_can_remove_staged_archive(self):
        stub = ProviderStub()
        result = self.call(stub)
        self.assertTrue(result["provider_readback_verified"])
        self.assertEqual(result["provider_member_hashes_verified"], 2)
        self.assertTrue(result["host_deleted"])
        self.assertFalse(self.bundle.exists())
        self.assertTrue(any(p.name.startswith("h67-private-")
                            for p in (self.root / "receipts").iterdir()))

    def test_partial_bytes_and_index_only_never_deletes_source(self):
        stub = ProviderStub(weak=True)
        with self.assertRaisesRegex(ValueError, "every uploaded member"):
            self.call(stub)
        self.assertTrue(self.bundle.is_file())

    def test_public_provider_state_never_proceeds_to_readback(self):
        stub = ProviderStub(public=True)
        with self.assertRaisesRegex(ValueError, "private provider"):
            self.call(stub)
        self.assertFalse(any(k == "readback" for k, _ in stub.calls))
        self.assertTrue(self.bundle.is_file())

    def test_no_token_never_calls_provider(self):
        stub = ProviderStub()
        with self.assertRaisesRegex(ValueError, "token required"):
            self.call(stub, token="")
        self.assertFalse(stub.calls)
        self.assertTrue(self.bundle.is_file())

    def test_archive_digest_corruption_blocks_external_api(self):
        stub = ProviderStub()
        with self.assertRaisesRegex(ValueError, "pinned SHA"):
            publish_private(self.bundle, "f"*64, GIT, TOKEN,
                            self.root / "receipts", provider=stub)
        self.assertFalse(stub.calls)

    def test_colab_upload_does_not_create_session_or_start_training(self):
        fake = mock.Mock(return_value=mock.Mock(returncode=0))
        hardware = mock.Mock(return_value="T4")
        result = upload(self.bundle, "hala-h67", self.result["sha256"], GIT,
                        runner=fake, hardware=hardware)
        self.assertEqual(result["status"], "UPLOADED_UNVERIFIED_REMOTE_BYTES")
        self.assertFalse(result["remote_integrity_verified"])
        self.assertEqual(fake.call_args.args[0][0:2], ["colab", "upload"])
        hardware.assert_called_once_with("hala-h67")

    def test_colab_upload_refuses_cpu_before_provider_upload(self):
        fake = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "approved T4"):
            upload(self.bundle, "hala-h67", self.result["sha256"], GIT,
                   runner=fake, hardware=lambda _: "CPU")
        fake.assert_not_called()

    def test_stage_upload_wrong_digest_refused_without_provider(self):
        fake = mock.Mock()
        hw = mock.Mock()
        with self.assertRaisesRegex(ValueError, "reviewed digest"):
            upload(self.bundle, "hala-h67", "f"*64, GIT,
                   runner=fake, hardware=hw)
        hw.assert_not_called()
        fake.assert_not_called()


if __name__ == "__main__":
    unittest.main()
