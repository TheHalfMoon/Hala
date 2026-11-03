"""H387: original owner processor cache cannot substitute HF-loaded bytes.

All eight fake assets are tiny invented content under TemporaryDirectory.
No model downloads, weights, private screenshots, real gold, GPU or training.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import unittest

from h218.pinned_qwen_processor_bytes import (
    PINNED_PUBLIC_FILES, SNAPSHOT_REVISION, verify_files,
)
from h387.verified_processor_handoff import verified_processor_handoff


def fabricated_cache(base: Path):
    source = base / SNAPSHOT_REVISION
    source.mkdir()
    invented = {}
    for index, name in enumerate(sorted(PINNED_PUBLIC_FILES)):
        raw = ("INVENTED-PUBLIC-CACHE-TEST-%02d-%s" %
               (index, name)).encode("ascii")
        (source / name).write_bytes(raw)
        invented[name] = (len(raw), sha256(raw).hexdigest())
    return source, invented


def fake_verifier(expected):
    return lambda path: verify_files(path, expected)


class H387ProcessorStagingTests(unittest.TestCase):
    def test_only_exact_eight_staged_bytes_reach_loader(self):
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            before = set(source.parent.iterdir())
            with (
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=fake_verifier(pinned)),
            ):
                with verified_processor_handoff(source) as staged:
                    self.assertNotEqual(staged, source)
                    self.assertEqual(staged.name, SNAPSHOT_REVISION)
                    self.assertEqual({p.name for p in staged.iterdir()},
                                     set(pinned))
                    self.assertEqual(verify_files(staged, pinned)
                                     ["verified_files"], 8)
                    stage_parent = staged.parent
                    # Original owner cache changes AFTER the verified copy:
                    # HF must still read the exact unchanged staged bytes.
                    original = source / "config.json"
                    original.write_bytes(b"X" * pinned["config.json"][0])
                    self.assertEqual(sha256((staged / "config.json")
                                            .read_bytes()).hexdigest(),
                                     pinned["config.json"][1])
                self.assertFalse(stage_parent.exists())
            self.assertEqual(set(source.parent.iterdir()), before)

    def test_corrupt_source_after_initial_validation_refuses_copy(self):
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            before = set(source.parent.iterdir())
            calls = []
            def verify_then_corrupt(path):
                receipt = verify_files(path, pinned)
                calls.append(path)
                if len(calls) == 1:
                    file = source / "config.json"
                    file.write_bytes(b"Z" * pinned["config.json"][0])
                return receipt

            with (
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=verify_then_corrupt),
            ):
                with self.assertRaisesRegex(
                    ValueError, "H387_PINNED_PROCESSOR_BYTES_NOT_STABLE"
                ):
                    with verified_processor_handoff(source):
                        self.fail("altered cache must never reach loader")
            self.assertEqual(calls, [source])
            self.assertEqual(set(source.parent.iterdir()), before)

    def test_extra_file_is_denied_before_processor_and_no_staging_remains(self):
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            (source / "invented-extra.safetensors").write_bytes(b"fake")
            with (
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=fake_verifier(pinned)),
            ):
                with self.assertRaises(ValueError):
                    with verified_processor_handoff(source):
                        self.fail("unexpected file must refuse")
            self.assertEqual(len(list(source.parent.iterdir())), 1)

    def test_staged_bytes_removed_even_if_consumer_raises(self):
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            with (
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=fake_verifier(pinned)),
            ):
                with self.assertRaisesRegex(RuntimeError, "invented abort"):
                    with verified_processor_handoff(source) as staged:
                        staged_parent = staged.parent
                        raise RuntimeError("invented abort")
            self.assertFalse(staged_parent.exists())
            self.assertEqual(len(list(source.parent.iterdir())), 1)

    def test_actual_h216_loader_interface_never_uses_mutable_original(self):
        from h216.real_qwen_processor_alignment import run_real_processor_audit
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            seen = []
            class FakeQwenProcessor:
                # H216 validates the API before the separate fake audit.
                def apply_chat_template(self, *args, **kwargs):
                    raise AssertionError('no fake tokenization allowed')
                def batch_decode(self, *args, **kwargs):
                    raise AssertionError('no fake decoding allowed')

            def fake_load(path, **kwargs):
                staged = Path(path)
                self.assertNotEqual(staged, source)
                self.assertEqual(kwargs, {
                    "local_files_only": True, "trust_remote_code": False,
                })
                self.assertEqual(verify_files(staged, pinned)["verified_files"],8)
                seen.append(staged)
                # Change owner cache between HF loading and subsequent audit.
                (source / "config.json").write_bytes(b"B" * pinned["config.json"][0])
                self.assertEqual(verify_files(staged, pinned)["verified_files"],8)
                return FakeQwenProcessor()

            fake_transformers = ModuleType("transformers")
            fake_transformers.Qwen3VLProcessor = FakeQwenProcessor
            fake_transformers.AutoProcessor = SimpleNamespace(
                from_pretrained=fake_load)
            with (
                patch.dict(sys.modules, {"transformers": fake_transformers}),
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=fake_verifier(pinned)),
                patch("h216.real_qwen_processor_alignment.verify_cached_processor_root"),
                patch("h216.real_qwen_processor_alignment.verify_checked_in_source",
                      return_value={
                          "candidate_cases_checked": 96,
                          "rgba_sha_matches": 96,
                          "official_train_admitted": False,
                      }),
                patch("h216.real_qwen_processor_alignment.audit_sample",
                      return_value={"prompt_tokens": 6, "supervised_tokens": 7,
                                    "vision_tensor_elements": 4}),
            ):
                receipts = run_real_processor_audit(source)
            self.assertEqual(len(receipts), 4)
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].parent.exists())

    def test_actual_h361_loader_interface_never_uses_mutable_original(self):
        from h361.original_qwen_processor import (
            run_pinned_offline_candidate_processor,
        )
        with TemporaryDirectory() as tmp:
            source, pinned = fabricated_cache(Path(tmp))
            captured = []
            class FakeQwenProcessor:
                pass

            def fake_load(path, **kwargs):
                staged = Path(path)
                captured.append(staged)
                self.assertNotEqual(staged, source)
                self.assertEqual(verify_files(staged, pinned)["verified_files"], 8)
                self.assertEqual(kwargs, {
                    "local_files_only": True, "trust_remote_code": False,
                })
                return FakeQwenProcessor()

            def fake_source_rows(proc, root, family):
                self.assertEqual(type(proc), FakeQwenProcessor)
                return (
                    [{"language": "en", "kind": "click"}] * 2 +
                    [{"language": "ar", "kind": "click"}] * 2
                    if family == "controlgarden-single-app" else
                    [{"language": "en", "kind": "click"}] * 3 +
                    [{"language": "ar", "kind": "click"}] * 3 +
                    [{"language": "en", "kind": "type_text"}] +
                    [{"language": "ar", "kind": "type_text"}]
                )
            fake_transformers = ModuleType("transformers")
            fake_transformers.Qwen3VLProcessor = FakeQwenProcessor
            fake_transformers.AutoProcessor = SimpleNamespace(
                from_pretrained=fake_load)
            datasets = {
                "controlgarden-single-app": Path(tmp),
                "noterelay-original-multistep": Path(tmp),
            }
            with (
                patch.dict(sys.modules, {"transformers": fake_transformers}),
                patch("h387.verified_processor_handoff.PINNED_PUBLIC_FILES",
                      pinned),
                patch("h387.verified_processor_handoff.verify_pinned_snapshot",
                      side_effect=fake_verifier(pinned)),
                patch("h361.original_qwen_processor._candidate_rows",
                      return_value=[]),
                patch("h361.original_qwen_processor.verify_cached_processor_root"),
                patch("h361.original_qwen_processor.real_processor_messages",
                      side_effect=fake_source_rows),
            ):
                receipt = run_pinned_offline_candidate_processor(
                    source, datasets)
            self.assertTrue(receipt["processor_image_and_prompt_parity"])
            self.assertFalse(receipt["model_weights_loaded"])
            self.assertFalse(receipt["can_train"])
            self.assertEqual(len(captured), 1)
            self.assertFalse(captured[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
