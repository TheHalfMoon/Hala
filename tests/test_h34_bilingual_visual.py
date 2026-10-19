import hashlib
import builtins
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import zlib
from contextlib import nullcontext
from types import SimpleNamespace

from h34.audit import audit
from h34.corpus import build_cases, canonical_cases, case_sha256
from h34.geometry import bind_point
from h34.protocol import parse_raw, score_attempt, summarize
from h34.render import load_font_pin, fetch_pinned_font
from h34.runner import owned_process, source_identity, _worker, run, save_json
from tools.build_h34_kaggle import generate, FILES, KERNEL_ID


def png_bytes(case, shade=240):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", case["width"], case["height"], 8, 2, 0, 0, 0)
    pixels = (b"\0" + bytes([shade]) * case["width"] * 3) * case["height"]
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def evidence(output, count=1, error=None):
    """Engineering fixture only: these blank pixels do not qualify the renderer/model."""
    pin = load_font_pin()
    cases = build_cases()
    (output / "cases.sha256").write_bytes((case_sha256() + "\n").encode("ascii"))
    images, rows = [], []
    for index, case in enumerate(cases[:count]):
        payload = png_bytes(case, 200 + index)
        name = case["id"] + ".png"
        (output / name).write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        images.append(dict(case_id=case["id"], path=name, sha256=sha, bytes=len(payload),
                           width=case["width"], height=case["height"], font_sha256=pin["sha256"],
                           font_revision=pin["upstream_revision"], license=pin["license"],
                           pillow="12.3.0", raqm="0.10.5", renderer="hala-h34-fixed-v1"))
        rows.append(dict(case_id=case["id"], language=case["language"], image_sha256=sha,
                         raw=None if error else '{"abstain":true}', error=error))
    (output / "image_manifest.json").write_text(json.dumps(images), encoding="utf-8")
    raw = output / "raw_attempts.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(raw.read_bytes()).hexdigest(), rows, images


class CorpusTests(unittest.TestCase):
    def test_case_counts_and_target_is_not_positional_rule(self):
        cases = build_cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual({lang: sum(c["language"] == lang for c in cases) for lang in ("en", "ar")},
                         {"en": 16, "ar": 16})
        self.assertEqual({layout: sum(c["layout"] == layout for c in cases)
                          for layout in ("row", "column", "rtl_row", "scattered")},
                         {"row": 8, "column": 8, "rtl_row": 8, "scattered": 8})
        for case in cases:
            self.assertEqual(sum(c["is_expected"] for c in case["controls"]), 1)
            expected = next(c for c in case["controls"] if c["is_expected"])
            self.assertEqual(expected["label"], case["expected_label"])
        self.assertEqual(len({next(c["control_index"] for c in x["controls"] if c["is_expected"])
                              for x in cases}), 4)
        for domain in {case["domain"] for case in cases}:
            positions = {next(c["control_index"] for c in case["controls"] if c["is_expected"])
                         for case in cases if case["domain"] == domain and case["language"] == "en"}
            self.assertGreater(len(positions), 1)

    def test_canonical_case_hash_is_stable(self):
        path = Path("h34/frozen/cases.json")
        self.assertEqual(path.read_bytes(), canonical_cases())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case_sha256())


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.case = next(c for c in build_cases() if c["id"] == "h34-mail-en-row")

    def test_wrong_unique_target_is_valid_point_but_semantically_wrong(self):
        target = next(x for x in self.case["controls"] if not x["is_expected"])
        x, y, w, h = target["bounds"]
        result = score_attempt(self.case, json.dumps({"point": [(x+w/2)/960, (y+h/2)/540]}))
        self.assertTrue(result["parse_valid"])
        self.assertTrue(result["gate_valid_unique"])
        self.assertTrue(result["wrong_target"])
        self.assertFalse(result["target_hit"])

    def test_half_open_edges_and_full_observed_uniqueness(self):
        a = {"bounds": [0, 0, 20, 20]}
        b = {"bounds": [20, 0, 20, 20]}
        self.assertIs(bind_point((20, 10), [a, b], [a, b], 40, 20), b)
        excluded = {"bounds": [5, 5, 20, 20]}
        eligible = {"bounds": [10, 10, 20, 20]}
        self.assertIsNone(bind_point((15, 15), [excluded, eligible], [eligible], 40, 40))
        self.assertIs(bind_point((25, 25), [excluded, eligible], [eligible], 40, 40), eligible)

    def test_layout_swaps_use_case_geometry(self):
        row = next(c for c in build_cases() if c["id"] == "h34-calendar-en-row")
        column = next(c for c in build_cases() if c["id"] == "h34-calendar-en-column")
        row_target = next(x for x in row["controls"] if x["is_expected"])
        col_target = next(x for x in column["controls"] if x["is_expected"])
        self.assertNotEqual(row_target["bounds"], col_target["bounds"])

    def test_parser_rejects_bad_numeric_and_duplicate_values(self):
        invalid = ['{"point":[true,0.1]}', '{"point":[NaN,0.1]}',
                   '{"point":[0.2,0.3],"point":[0.4,0.5]}',
                   '{"point":[0.2,0.3],"extra":1}', '{"abstain":false}',
                   '{"point":[1.01,0.1]}']
        invalid.append('{"point":[' + '9' * 1500 + ',0.1]}')
        invalid.append('[' * 1000 + '0' + ']' * 1000)
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_raw(raw, 960, 540)
        self.assertIsNone(parse_raw('{"abstain":true}', 960, 540))

    def test_error_partial_and_tampered_raw_audit(self):
        cases = build_cases()
        self.assertEqual(summarize(cases, [score_attempt(c, None, "timeout") for c in cases])
                         ["languages"]["ar"]["error"], 16)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            raw = output/"raw_attempts.jsonl"
            digest, _, _ = evidence(output)
            result = audit(output, digest)
            self.assertEqual(result["attempted_records"], 1)
            self.assertEqual(result["metrics"]["languages"]["en"]["denominator"], 16)
            raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                audit(output, digest)


class EvidenceBoundaryTests(unittest.TestCase):
    def test_partial_bilingual_audit_counts_and_geometry_are_independently_known(self):
        # Coordinates come directly from the fixed case layout formulas:
        # mail/en/row Archive = center (164, 332); mail/en/column Delete = (480, 282).
        # mail/ar/row إرسال = (164, 332); its next control حذف = (380, 332).
        cases = build_cases()
        self.assertEqual([c["id"] for c in cases[:8]], [
            "h34-mail-en-row", "h34-mail-en-column", "h34-mail-en-rtl_row",
            "h34-mail-en-scattered", "h34-mail-ar-row", "h34-mail-ar-column",
            "h34-mail-ar-rtl_row", "h34-mail-ar-scattered",
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            digest, rows, _ = evidence(output, count=8)
            responses = [
                '{"point":[0.17083333333333334,0.6148148148148148]}',  # known EN correct
                '{"point":[0.5,0.37037037037037035]}',              # EN wrong: Send
                '{"abstain":true}',
                '{"point":[NaN,0.1]}',                            # malformed JSON value
                '{"point":[0.17083333333333334,0.6148148148148148]}',  # known AR correct
                '{"point":[0.5,0.37037037037037035]}',              # AR wrong: another control
                '',                                                 # empty response
                None,
            ]
            for index, (row, raw) in enumerate(zip(rows, responses)):
                row["raw"] = raw
                row["error"] = "RuntimeError" if index == 7 else None
            raw_path = output / "raw_attempts.jsonl"
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()

            result = audit(output, digest)
            metrics = result["metrics"]
            self.assertEqual(result["attempted_records"], 8)
            self.assertFalse(result["complete"])
            self.assertEqual(metrics["denominator"], 32)
            self.assertEqual(metrics["languages"]["en"], {
                "parse_valid": 3, "target_hit": 1, "wrong_target": 1,
                "abstain": 1, "error": 12, "gate_valid_unique": 2,
                "denominator": 16, "observed_control_count": 64,
            })
            self.assertEqual(metrics["languages"]["ar"], {
                "parse_valid": 2, "target_hit": 1, "wrong_target": 1,
                "abstain": 0, "error": 13, "gate_valid_unique": 2,
                "denominator": 16, "observed_control_count": 64,
            })

            # Anchor positive semantics to explicit source geometry, independent of the audit counts.
            en_row = cases[0]
            en_hit = score_attempt(en_row, responses[0])
            self.assertTrue(en_hit["target_hit"])
            self.assertTrue(en_hit["gate_valid_unique"])
            ar_row = cases[4]
            ar_hit = score_attempt(ar_row, responses[4])
            self.assertTrue(ar_hit["target_hit"])
            self.assertTrue(ar_hit["gate_valid_unique"])

            # Reusing the old row coordinate on the changed column layout misses every control;
            # the fresh column target center is a valid unique hit.
            column = cases[1]
            self.assertFalse(score_attempt(column, responses[0])["target_hit"])
            fresh_column_point = '{"point":[0.5,0.5222222222222223]}'
            fresh_hit = score_attempt(column, fresh_column_point)
            self.assertTrue(fresh_hit["target_hit"])
            self.assertTrue(fresh_hit["gate_valid_unique"])

    def test_manifest_and_png_tampering_fail_with_unchanged_raw_anchor(self):
        for mutation in ("swap_png", "swap_hash", "swap_path", "dimensions", "pixel_decode", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                digest, _, images = evidence(output, count=2)
                first, second = (output / row["path"] for row in images)
                if mutation == "swap_png":
                    a, b = first.read_bytes(), second.read_bytes()
                    first.write_bytes(b)
                    second.write_bytes(a)
                elif mutation == "swap_hash":
                    images[0]["sha256"], images[1]["sha256"] = images[1]["sha256"], images[0]["sha256"]
                elif mutation == "swap_path":
                    images[0]["path"], images[1]["path"] = images[1]["path"], images[0]["path"]
                elif mutation == "dimensions":
                    images[0]["width"] = 1
                elif mutation == "pixel_decode":
                    first.write_bytes(first.read_bytes()[:-1])
                    images[0].update(sha256=hashlib.sha256(first.read_bytes()).hexdigest(), bytes=first.stat().st_size)
                else:
                    first.unlink()
                (output / "image_manifest.json").write_text(json.dumps(images), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit(output, digest)

    def test_valid_errors_and_missing_cases_keep_unconditional_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            digest, _, _ = evidence(output, count=2, error="RuntimeError")
            result = audit(output, digest)
            self.assertFalse(result["complete"])
            self.assertEqual(result["attempted_records"], 2)
            self.assertEqual(result["metrics"]["denominator"], 32)
            self.assertEqual(result["metrics"]["languages"]["en"]["error"], 16)
            self.assertEqual(result["metrics"]["languages"]["ar"]["error"], 16)
            self.assertEqual(result["raw_attempts"][0]["image_sha256"],
                             hashlib.sha256((output / "h34-mail-en-row.png").read_bytes()).hexdigest())

    def test_strict_raw_schema_language_order_and_types(self):
        for mutation in ("order", "language", "extra", "missing", "raw_type", "error_type", "error_and_raw", "duplicate", "blank", "partial"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                _, rows, _ = evidence(output, count=2)
                if mutation == "order": rows.reverse()
                elif mutation == "language": rows[0]["language"] = "ar"
                elif mutation == "extra": rows[0]["claimed_success"] = True
                elif mutation == "missing": rows[0].pop("image_sha256")
                elif mutation == "raw_type": rows[0]["raw"] = {"abstain": True}
                elif mutation == "error_type": rows[0].update(raw=None, error=True)
                elif mutation == "error_and_raw": rows[0]["error"] = "RuntimeError"
                payload = "".join(json.dumps(row) + "\n" for row in rows)
                if mutation == "duplicate": payload = payload.replace('"language": "en"', '"language": "en", "language": "en"', 1)
                elif mutation == "blank": payload += "\n"
                elif mutation == "partial": payload = payload.rstrip("\n")
                raw = output / "raw_attempts.jsonl"
                raw.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit(output, hashlib.sha256(raw.read_bytes()).hexdigest())

    def test_font_cache_and_license_match_independent_upstream_receipt(self):
        pin = load_font_pin()
        self.assertEqual(pin["archive_sha256"], "5c6e497a2f36552cb5ffb112c413a6af39c0f3c47653662b90b4fa6499822fd7")
        self.assertEqual(pin["sha256"], "7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954")
        self.assertEqual(hashlib.sha256(Path(pin["license_file"]).read_bytes()).hexdigest(), pin["license_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "font.ttf"
            path.write_bytes(b"not the pinned font")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), self.assertRaises(RuntimeError):
                fetch_pinned_font(path, pin)


class BootstrapBoundaryTests(unittest.TestCase):
    def bootstrap_fixture(self, directory, authorized=True):
        root = Path(directory) / "source"
        root.mkdir()
        # Frozen independent source fixture; fake clone copies these exact bytes.
        for name in FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(__file__).resolve().parent.parent / name).read_bytes())
        files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
        inv = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        bundle = Path(directory) / "bundle"
        sha = "a" * 40
        with patch("tools.build_h34_kaggle.inventory", return_value=inv):
            generate(root, sha, bundle, research_allowed=authorized)
        notebook = json.loads((bundle / "hala-h34-bilingual-visual-dev.ipynb").read_text(encoding="utf-8"))
        working = Path(directory) / "working"
        working.mkdir()
        def remote_path(*parts):
            if str(parts[0]).startswith("/kaggle/working"):
                return Path(str(parts[0]).replace("/kaggle/working", str(working), 1), *parts[1:])
            return Path(*parts)
        clones = []
        def fake_clone(args, **kwargs):
            self.assertEqual(args[:2], ["git", "clone"])
            self.assertFalse((working / "hala-h34-dev-out").exists())
            destination = Path(args[-1])
            self.assertFalse(destination.exists())
            shutil.copytree(root, destination)
            clones.append(destination)
        def fake_git(args, **kwargs):
            self.assertEqual(args[0], "git")
            if args[1:] == ["rev-parse", "HEAD"]: return sha + "\n"
            if args[1:] == ["status", "--porcelain", "--untracked-files=all"]: return ""
            raise AssertionError(args)
        return root, working, bundle, notebook, remote_path, fake_clone, fake_git, clones

    def execute_bootstrap(self, fixture):
        _, _, _, notebook, remote_path, fake_clone, fake_git, _ = fixture
        original_import = builtins.__import__
        def bootstrap_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pathlib" and "Path" in fromlist:
                return SimpleNamespace(Path=remote_path)
            return original_import(name, globals, locals, fromlist, level)
        bootstrap_builtins = builtins.__dict__.copy()
        bootstrap_builtins["__import__"] = bootstrap_import
        with patch("platform.system", return_value="Linux"), \
             patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, clear=False), \
             patch("subprocess.run", side_effect=fake_clone), patch("subprocess.check_output", side_effect=fake_git):
            exec("".join(notebook["cells"][1]["source"]), {"__builtins__": bootstrap_builtins})

    def test_actual_bootstrap_admits_before_clone_and_has_exact_private_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.bootstrap_fixture(directory)
            self.execute_bootstrap(fixture)
            self.assertEqual(len(fixture[-1]), 1)
            metadata = json.loads((fixture[2] / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["title"], KERNEL_ID.split("/")[1])
            with self.assertRaises(FileExistsError): self.execute_bootstrap(fixture)
            self.assertEqual(len(fixture[-1]), 1)

    def test_actual_bootstrap_denies_optin_existing_output_and_changed_source(self):
        for reason in ("optin", "prior_output", "changed_bytes"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                fixture = self.bootstrap_fixture(directory, authorized=reason != "optin")
                if reason == "prior_output": (fixture[1] / "hala-h34-dev-out").mkdir()
                elif reason == "changed_bytes": (fixture[0] / "h34/runner.py").write_bytes(b"changed after source inventory")
                with self.assertRaises((RuntimeError, FileExistsError)): self.execute_bootstrap(fixture)
                self.assertEqual(len(fixture[-1]), 1 if reason == "changed_bytes" else 0)

    def test_notebook_case_hash_uses_inventory_snapshot_after_working_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.bootstrap_fixture(directory)
            root, _, _, _, _, _, _, _ = fixture
            case_path = root / "h34/frozen/cases.json"
            original_read_bytes = Path.read_bytes
            observed_case_reads = []

            def tracked_read_bytes(path):
                if path == case_path:
                    observed_case_reads.append(path)
                    raise AssertionError("builder reread mutable case file after inventory")
                return original_read_bytes(path)

            inv = {"files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in FILES}}
            inv["sha256"] = hashlib.sha256(json.dumps(
                inv["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()

            def inventory_then_mutate(*_args):
                case_path.write_bytes(b"post-inventory working-tree mutation")
                return inv

            output = Path(directory) / "race-bundle"
            with patch("tools.build_h34_kaggle.inventory", side_effect=inventory_then_mutate), \
                 patch.object(Path, "read_bytes", autospec=True, side_effect=tracked_read_bytes):
                generate(root, "a" * 40, output)
            notebook = json.loads((output / "hala-h34-bilingual-visual-dev.ipynb").read_text(encoding="utf-8"))
            self.assertEqual(notebook["metadata"]["hala_case_sha256"], inv["files"]["h34/frozen/cases.json"])
            self.assertNotEqual(hashlib.sha256(original_read_bytes(case_path)).hexdigest(),
                                notebook["metadata"]["hala_case_sha256"])
            self.assertEqual(observed_case_reads, [])

    def test_runner_independently_checks_source_bytes_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.bootstrap_fixture(directory)
            root = fixture[0]
            originals = {name: (root / name).read_bytes() for name in FILES}
            def git(args, **kwargs):
                if args[1:] == ["rev-parse", "HEAD"]: return ("a" * 40).encode() + b"\n"
                if args[1] == "status": return b""
                if args[1] == "show": return originals[args[2].split(":", 1)[1]]
                raise AssertionError(args)
            with patch("h34.runner.__file__", str(root / "h34/runner.py")), patch("subprocess.check_output", side_effect=git):
                self.assertEqual(source_identity("a" * 40)["h34/render.py"], hashlib.sha256(originals["h34/render.py"]).hexdigest())
                (root / "h34/render.py").write_bytes(b"renderer changed")
                with self.assertRaisesRegex(ValueError, "source bytes"):
                    source_identity("a" * 40)


class WorkerBoundaryTests(unittest.TestCase):
    def test_source_failure_is_retained_without_loading_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_json(output / "run_metadata.json", {"state": "starting"})
            (output / "raw_attempts.jsonl").touch()
            with patch("h34.runner.source_identity", side_effect=ValueError("changed")), \
                 patch("h34.runner._load_model", side_effect=AssertionError("must not allocate")) as load:
                _worker(output, "a" * 40)
            load.assert_not_called()
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["state"], "incomplete")
            self.assertEqual(metadata["failure_type"], "ValueError")
            self.assertEqual((output / "raw_attempts.jsonl").read_bytes(), b"")

    def test_real_worker_pipeline_binds_images_excludes_oracle_and_retains_inference_error_hash(self):
        # All ML/renderer objects are offline engineering fakes, never qualification.
        import h34.runner as runner
        pin = load_font_pin()
        captured = []
        pixel_marker = object()
        class FakePNG:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def convert(self, mode): return pixel_marker
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                captured.append(messages)
                return "image-bearing prompt"
            def __call__(self, **kwargs):
                assert kwargs["images"] == [pixel_marker]
                class Inputs(dict):
                    input_ids = SimpleNamespace(shape=(1, 3))
                    def to(self, device): return self
                return Inputs()
        model = SimpleNamespace(generate=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline injected failure")))
        gpu = SimpleNamespace(uuid="GPU-offline-fixture", total_memory=16_000_000_000)
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: "Tesla T4 fixture",
                                                     get_device_properties=lambda i: gpu), inference_mode=nullcontext)
        pil = SimpleNamespace(Image=SimpleNamespace(open=lambda p: FakePNG()), features=SimpleNamespace(version_feature=lambda name: "0.10.5"))
        def render(case, path, font, font_pin):
            payload = png_bytes(case)
            path.write_bytes(payload)
            return dict(path=path.name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                        width=case["width"], height=case["height"], font_sha256=pin["sha256"],
                        font_revision=pin["upstream_revision"], license=pin["license"], pillow="12.3.0", raqm="0.10.5", renderer="hala-h34-fixed-v1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_json(output / "run_metadata.json", {"state": "starting"})
            (output / "raw_attempts.jsonl").touch()
            def load():
                metadata = json.loads((output / "run_metadata.json").read_text())
                self.assertEqual(metadata["packages"], runner.PINS)
                self.assertEqual(metadata["gpus"][0]["uuid"], "GPU-offline-fixture")
                self.assertEqual(metadata["raqm"], "0.10.5")
                self.assertEqual(len(json.loads((output / "image_manifest.json").read_text())), 32)
                self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
                return Processor(), model
            with patch.dict(sys.modules, {"torch": torch, "PIL": pil}), \
                 patch("h34.runner.source_identity", return_value={"frozen": "source"}), \
                 patch("h34.runner.importlib.metadata.version", side_effect=lambda name: runner.PINS[name]), \
                 patch("h34.runner.subprocess.check_output", return_value="0,Tesla T4 fixture,GPU-offline-fixture,16000\n"), \
                 patch("h34.runner.fetch_pinned_font", return_value=Path("offline-font")), \
                 patch("h34.runner.render_case", side_effect=render), patch("h34.runner._load_model", side_effect=load), patch("builtins.print"):
                _worker(output, "a" * 40)
            raw = output / "raw_attempts.jsonl"
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(row["error"] == "RuntimeError" and row["image_sha256"] for row in rows))
            self.assertEqual(len(captured), 32)
            for case, messages in zip(build_cases(), captured):
                self.assertEqual(messages, [{"role": "user", "content": [
                    {"type": "image", "image": pixel_marker}, {"type": "text", "text": case["goal"] + "\n" + runner.PROMPT}]}])
                self.assertNotIn("bounds", messages[0]["content"][1]["text"])
            result = audit(output, hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertEqual(result["metrics"]["languages"]["en"]["error"], 16)
            self.assertEqual(result["metrics"]["languages"]["ar"]["error"], 16)

    def test_supervisor_timeout_keeps_only_attempted_prefix_and_pending_image_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            def timeout(command, deadline):
                self.assertEqual(deadline, 1)
                _, rows, _ = evidence(output, count=2)
                raw = output / "raw_attempts.jsonl"
                raw.write_bytes((json.dumps(rows[0]) + "\n").encode())
                rows[1].update(raw=None, error="inference_interrupted")
                save_json(output / "pending_attempt.json", rows[1])
                return None, True
            with patch("h34.runner.admit"), patch("h34.runner.owned_process", side_effect=timeout):
                metadata = run(output, "a" * 40, timeout_seconds=1)
            self.assertEqual(metadata["state"], "incomplete")
            self.assertEqual(metadata["attempted_count"], 2)
            self.assertTrue(metadata["deadline_exceeded"])
            result = audit(output, metadata["raw_sha256"])
            self.assertEqual(result["attempted_records"], 2)
            self.assertEqual(result["metrics"]["languages"]["en"]["abstain"], 1)
            self.assertEqual(result["metrics"]["languages"]["en"]["error"], 15)
            self.assertFalse(result["complete"])


class DeadlineBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "actual owned process-group test runs separately in offline WSL Linux")
    def test_actual_deadline_kills_owned_parent_and_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pidfile = directory / "pids.json"
            late = directory / "late-write"
            child = "import time; from pathlib import Path; time.sleep(2); Path(" + repr(str(late)) + ").write_text('orphan')"
            parent = ("import os,subprocess,sys,json,time; from pathlib import Path; "
                      "p=subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); "
                      "Path(" + repr(str(pidfile)) + ").write_text(json.dumps([os.getpid(),p.pid])); time.sleep(20)")
            started = time.monotonic()
            code, timed_out = owned_process([sys.executable, "-c", parent], 0.5)
            self.assertTrue(timed_out)
            self.assertIsNone(code)
            self.assertLess(time.monotonic() - started, 3)
            pids = json.loads(pidfile.read_text())
            for pid in pids:
                stat = Path(f"/proc/{pid}/stat")
                self.assertFalse(stat.exists(), "owned process must be dead and reaped")
            time.sleep(2)
            self.assertFalse(late.exists())


if __name__ == "__main__":
    unittest.main()
