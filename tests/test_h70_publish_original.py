"""H70 owner-private uploader: mocks never contact external Kaggle or GPUs."""
import hashlib
import io
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path

from h70 import publish_original as p

GIT = "f" * 40


class FakeKaggle:
    def __init__(self, source: Path, domain: str, *, corrupt=False, incomplete=False):
        self.source = source
        self.domain = domain
        self.name = f"hala-h70-{domain}-2500.hala.bin"
        self.corrupt = corrupt
        self.incomplete = incomplete
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[:3] == ["kaggle", "datasets", "create"]:
            self.stage = Path(command[command.index("-p")+1])
            assert (self.stage / self.name).read_bytes() == self.source.read_bytes()
            assert not ("-u" in command or "--public" in command)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["kaggle", "datasets", "files"]:
            return subprocess.CompletedProcess(
                command, 0,
                f"name,size,creationDate\n{self.name},{self.source.stat().st_size},2026-10-23", "")
        if command[:3] == ["kaggle", "datasets", "download"]:
            path = Path(command[command.index("-p")+1]) / self.name
            data = self.source.read_bytes()
            if self.incomplete: data = data[:5]
            if self.corrupt: data = b"z" + data[1:]
            path.write_bytes(data)
            return subprocess.CompletedProcess(command, 0, "downloaded", "")
        raise AssertionError("unapproved CLI command")

def forbidden_anon(*args, **kwargs):
    raise urllib.error.HTTPError("https://www.kaggle.com/private",403,"forbidden",{},None)

class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.binary = self.root / "file.bin"
        self.binary.write_bytes(b"test private source archive with immutable bytes")
        self.hexsha = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.receipts = self.root / "proof"

    def tearDown(self):
        self.tmp.cleanup()

    def verified(self, path, sha, out, git_sha, domain):
        self.assertEqual(sha, self.hexsha)
        self.assertEqual(git_sha, GIT)
        self.assertEqual(domain, "sheet")
        if Path(path).read_bytes() != self.binary.read_bytes():
            raise ValueError("mock source corrupt")
        return {
            "status": "VERIFIED_CPU_SOURCE_RESTORED",
            "images": 2500, "actions": 19000,
            "labels_sha256": "b"*64, "source_revision": "c"*40
        }

    def test_complete_private_byte_readback_and_full_source_validation(self):
        provider=FakeKaggle(self.binary, "sheet")
        output=p.publish(self.binary,"sheet",self.hexsha,GIT,self.receipts,
                         cli=provider,opener=forbidden_anon,unpacker=self.verified)
        self.assertEqual(output["status"],"PRIVATE_BYTE_READBACK_VERIFIED")
        self.assertEqual(output["anonymous_http_status"],403)
        self.assertTrue(output["authenticated_full_byte_readback"])
        self.assertFalse(output["gpu_allocated"])
        self.assertTrue(self.binary.exists())
        self.assertEqual(len(provider.commands),3)
        self.assertIn("2500-private",output["provider_dataset"])
        self.assertTrue((self.receipts/"sheet-readback.json").is_file())

    def test_resume_existing_owner_dataset_without_duplicate_create(self):
        provider=FakeKaggle(self.binary, "sheet")
        output=p.publish(self.binary,"sheet",self.hexsha,GIT,self.receipts,
                         cli=provider,opener=forbidden_anon,
                         unpacker=self.verified,resume_existing=True)
        self.assertEqual(output["status"],"PRIVATE_BYTE_READBACK_VERIFIED")
        self.assertEqual([cmd[2] for cmd in provider.commands],["files","download"])
        self.assertTrue(self.binary.exists())

    def test_provider_index_delay_retries_without_duplicate_upload(self):
        original=FakeKaggle(self.binary,"sheet")
        attempts=[0]
        def flaky(command, **kwargs):
            if command[2]=="files" and attempts[0]<2:
                attempts[0]+=1
                return subprocess.CompletedProcess(command,1,"","not yet indexed")
            return original(command, **kwargs)
        delays=[]
        output=p.publish(self.binary,"sheet",self.hexsha,GIT,self.receipts,
                         cli=flaky,opener=forbidden_anon,
                         unpacker=self.verified,sleep=lambda sec:delays.append(sec))
        self.assertEqual(output["status"],"PRIVATE_BYTE_READBACK_VERIFIED")
        self.assertEqual(attempts[0],2)
        self.assertEqual(delays,[10,10])
        self.assertEqual(sum(cmd[2]=="create" for cmd in original.commands),1)

    def test_missing_or_modified_remote_bytes_cannot_admit_receipt(self):
        for args in ({"incomplete":True},{"corrupt":True}):
            with self.subTest(args=args):
                receipt=self.root/(("short" if "incomplete" in args else "corrupt")+"-receipt")
                provider=FakeKaggle(self.binary,"sheet",**args)
                with self.assertRaisesRegex(ValueError,"missing or short|byte SHA256"):
                    p.publish(self.binary,"sheet",self.hexsha,GIT,receipt,
                              cli=provider,opener=forbidden_anon,unpacker=self.verified)
                self.assertFalse((receipt/"sheet-readback.json").exists())
                self.assertTrue(self.binary.exists())

    def test_public_dataset_refused_without_auth_readback(self):
        provider=FakeKaggle(self.binary,"sheet")
        def public(*args,**kwargs):
            reply = io.BytesIO(b"public")
            reply.status = 200
            return reply
        with self.assertRaisesRegex(ValueError,"private dataset access"):
            p.publish(self.binary,"sheet",self.hexsha,GIT,self.receipts,
                      cli=provider,opener=public,unpacker=self.verified,
                      sleep=lambda seconds: None)
        self.assertEqual(len(provider.commands),2)
        self.assertFalse((self.receipts/"sheet-readback.json").exists())

    def test_unverified_input_never_contacts_external_provider(self):
        provider=FakeKaggle(self.binary,"sheet")
        with self.assertRaisesRegex(ValueError,"reviewed SHA256"):
            p.publish(self.binary,"sheet","f"*64,GIT,self.receipts,
                      cli=provider,opener=forbidden_anon,unpacker=self.verified)
        self.assertEqual(provider.commands,[])

    def test_existing_receipt_prevents_accidental_dataset_recreation(self):
        self.receipts.mkdir()
        (self.receipts/"sheet-readback.json").write_text('{"existing":true}')
        provider=FakeKaggle(self.binary,"sheet")
        with self.assertRaises(FileExistsError):
            p.publish(self.binary,"sheet",self.hexsha,GIT,self.receipts,
                      cli=provider,opener=forbidden_anon,unpacker=self.verified)
        self.assertEqual(provider.commands,[])

    def test_slider_is_reserved_for_previous_h69_work(self):
        provider=FakeKaggle(self.binary,"slider")
        with self.assertRaisesRegex(ValueError,"three not-yet-published"):
            p.publish(self.binary,"slider",self.hexsha,GIT,self.receipts,
                      cli=provider,opener=forbidden_anon,unpacker=self.verified)
        self.assertEqual(provider.commands,[])


if __name__ == "__main__":
    unittest.main()
