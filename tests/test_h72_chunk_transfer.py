"""H72 no-network tests for byte-perfect resumable stage transport."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h72 import chunk_transfer as host
from h72 import remote_stage as remote

GIT = "a" * 40


class ChunkTransferTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        self.source = self.root / "stage.zip"
        self.source.write_bytes(b"A" * remote.MAX_CHUNK + b"B" * 101)
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.parts = self.root / "colab"
        self.parts.mkdir()
        self.final = self.root / "remote-stage.zip"
        self.plan = host.plan(self.source, self.digest, GIT)
        (self.parts / "manifest.json").write_text(json.dumps(self.plan), "utf8")

    def tearDown(self):
        self.t.cleanup()

    def populate(self):
        content = self.source.read_bytes()
        offset = 0
        for p in self.plan["parts"]:
            (self.parts / p["name"]).write_bytes(content[offset:offset + p["bytes"]])
            offset += p["bytes"]

    def test_bounded_two_part_plan_and_content_hash(self):
        self.assertEqual(len(self.plan["parts"]), 2)
        self.assertEqual(self.plan["parts"][0]["bytes"], remote.MAX_CHUNK)
        self.assertEqual(self.plan["parts"][1]["bytes"], 101)
        self.assertEqual(remote.read_manifest(self.parts), self.plan)

    def test_atomic_remote_reassembly_matches_exact_sha(self):
        self.populate()
        result = remote.assemble(self.parts, self.final)
        self.assertEqual(result["status"], "H72_STAGE_READY")
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(remote.sha(self.final), self.digest)
        self.assertEqual(remote.assemble(self.parts, self.final), result)

    def test_changed_chunk_rejected_without_final_archive(self):
        self.populate()
        (self.parts / "chunk_00001.bin").write_bytes(b"X"*101)
        self.assertEqual(len(remote.present(self.plan, self.parts)), 1)
        with self.assertRaisesRegex(ValueError, "unverified or missing"):
            remote.assemble(self.parts, self.final)
        self.assertFalse(self.final.exists())

    def test_untrusted_manifest_path_not_accepted(self):
        self.plan["parts"][0]["name"] = "../outside"
        (self.parts / "manifest.json").write_text(json.dumps(self.plan), "utf8")
        with self.assertRaisesRegex(ValueError, "untrusted"):
            remote.read_manifest(self.parts)

    def test_wrong_original_sha_fails_before_colab_contact(self):
        with mock.patch.object(host, "executable", side_effect=AssertionError("provider touched")):
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                host.transfer(self.source, "f"*64, GIT, "t4-test", cpu_test=True)

    def test_resume_skips_verified_first_chunk_and_uploads_exact_second_bytes(self):
        first = self.plan["parts"][0]
        (self.parts / first["name"]).write_bytes(self.source.read_bytes()[:first["bytes"]])
        sent = []
        def fake_remote(cli, *args, **kwargs):
            if args[0] == "sessions":
                return "[hala-h72-test] cpu-standard | Hardware: CPU | Shape: Standard | Variant: CPU"
            if args[0] == "upload":
                name = Path(args[-1]).name
                src = Path(args[-2])
                shutil.copyfile(src, self.parts / name)
                sent.append(name)
                return "Uploaded successfully"
            if args[0] == "exec":
                action = next(s.split("=",1)[1] for s in args if s.startswith("H72_ACTION="))
                if action == "prepare":
                    return "H72_ROOT_PREPARED"
                if action == "probe":
                    return "H72_HAVE_JSON=" + json.dumps(remote.present(remote.read_manifest(self.parts), self.parts))
                if action == "assemble":
                    return "H72_STAGE_READY=" + json.dumps(remote.assemble(self.parts,self.final))
            raise AssertionError("unexpected provider invocation")
        with mock.patch.object(host,"_run",side_effect=fake_remote), mock.patch.object(host,"executable",return_value="colab"):
            result = host.transfer(self.source,self.digest,GIT,"hala-h72-test",cpu_test=True)
        self.assertEqual(result["status"],"REMOTE_ZIP_SHA_VERIFIED")
        self.assertEqual(result["resumed_verified_chunks"],1)
        self.assertIn("manifest.json",sent)
        self.assertIn("chunk_00001.bin",sent)
        self.assertNotIn("chunk_00000.bin",sent)
        self.assertEqual(remote.sha(self.final),self.digest)
        self.assertFalse(result["gpu_training_executed"])

    def test_refuse_changed_existing_completed_stage(self):
        self.populate()
        self.final.write_bytes(b"not the original dataset")
        with self.assertRaises(FileExistsError):
            remote.assemble(self.parts,self.final)


if __name__ == "__main__":
    unittest.main()
