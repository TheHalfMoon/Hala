"""Prospective H22 free-Kaggle Save & Run All batch operator.

This operator does not train, tune, merge, deploy or publish model artifacts.
It is deliberately distinct from the historical interactive attempt3 operator.
All three TEST modes must run with the same observed physical GPU in ONE batch.
"""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

SOURCE = "07591f109b6683c9302fcf890259ddf0bde91494"
FREEZE = "98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
ARTIFACT_COMMIT = "c7f2e16244910c7b4246749b3186d49fca5ce7b4"
DEV_SHA = "60a0986cc196041c2474314624e0f88c9c2986d75bc58cf0ddb44623ed294e02"
H19_SHA = "d6be1ffba54a3a44978da4c922130fc2d7e74bdc2532a14d0d1e7cc3a691cead"
H19_MANIFEST = "ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e"
ADAPTER_MANIFEST = "7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d"
EARLIER_FP16_ARCHIVE = "e098a7631708aa50609e56ba7701c01a1c5cc5804b87f9994b6dded181abf471"
EARLIER_FP16_COMMIT = "9568d58c74f3e1326b55b04d304f643e2bf1b99f"
ARCHIVE_BASE = "https://raw.githubusercontent.com/TheHalfMoon/Hala/"
MODES = ("fp16", "int8", "nf4")
ROOT = Path("/kaggle/working/hala-h22-batch")
CACHE = "/tmp/hala-h22-cache"
ADAPTER_FILES = {
    "adapter_config.json": "21bbc4037d9fe27e1281700e72dcc402c4ce801d2d27798ac64cba33ac4d5de2",
    "adapter_model.safetensors": "7eb2f17df11151e5b77a5c715c59f56c854ca7b0d0da3ee591f394b501826fa7",
    "training_metadata.json": "d1251b39c3d71fcd410ec0fce631b2c065a6df4978309bedaff168807314020d",
    "manifest.json": ADAPTER_MANIFEST,
}
ADAPTER_IDENTITY = {
    "base_model": "Qwen/Qwen3-VL-2B-Instruct", "base_revision": MODEL_REV,
    "data_sha256": "af540e4db2289540fb8adecd0665439c4d6055ffc86673a3286e902c86c0340e",
    "evaluation_sha256": "e9f98694089d5411d3a6c570c7de2ede8f82cbeb3132e47904d17301030c0768",
    "source_commit": "5b6eb4bf29acbaf21aa199f113f78af4f268b3b7",
}
DEV = (
    ("e252b7805055fa6a89e1d92b6bba7faa5dbf1efc", "6d5cfe7c3746a76eaceb8ebc37605d0d98cc69b69ca94765421f081b64d64580"),
    ("e3d400424766aad3c5856ba29266e11033392ef5", "e0dbe2d692c522943b3e0c49c198e540416cbabfa2b19499c9da4a518b8339e5"),
    ("d08f522de433d9da866ec3907161e4a9b326efba", "84dac2fd6ab44d751cefbe9ce61a958ef44a39aad3b1f09e880c5350bbd87458"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def public_artifact(commit, suffix, expected, maximum):
    """Read only immutable public artifacts; bounds and digest are mandatory."""
    if re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        raise ValueError("Immutable commit SHA required")
    if not suffix or suffix.startswith("/") or ".." in Path(suffix).parts:
        raise ValueError("Unsafe artifact path")
    url = ARCHIVE_BASE + commit + "/" + suffix
    with urllib.request.urlopen(url, timeout=45) as reply:
        if reply.geturl() != url:
            raise ValueError("Unexpected artifact redirect")
        raw = reply.read(maximum + 1)
    if len(raw) > maximum or digest(raw) != expected:
        raise ValueError("Public evidence digest or size mismatch: " + suffix)
    return raw


def safe_unpack(raw, target):
    """No traversal, links, duplicate members, ZIP bombs or replacement."""
    target = Path(target)
    if target.exists():
        raise FileExistsError("Existing evidence/dataset directory must not be replaced")
    target.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        if len(infos) > 512 or len({x.filename for x in infos}) != len(infos):
            raise ValueError("Unexpected ZIP member count or duplicates")
        if sum(x.file_size for x in infos) > 32 * 1024**2:
            raise ValueError("ZIP content budget exceeded")
        for item in infos:
            name = item.filename
            if (not name or name.startswith("/") or chr(92) in name or
                    any(x in ("", ".", "..") for x in name.rstrip("/").split("/")) or
                    item.file_size > 8 * 1024**2 or ((item.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe ZIP member")
            path = target / name
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def prepare(root=ROOT):
    """Network-only source/evidence preparation. No model download."""
    if os.name != "posix" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Must run in actual remote Kaggle Linux environment")
    if root.exists():
        raise FileExistsError("Batch workspace already exists; never overwrite old evidence")
    root.mkdir(parents=True)
    source = root / "source"
    subprocess.run(["git", "clone", "--quiet", "https://github.com/TheHalfMoon/Hala.git", str(source)],
                   check=True, timeout=240)
    subprocess.run(["git", "checkout", "--detach", SOURCE], cwd=source, check=True, timeout=60,
                   stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != SOURCE:
        raise ValueError("Historical source identity mismatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source, text=True).strip():
        raise ValueError("Historical source must be clean")
    safe_unpack(public_artifact(ARTIFACT_COMMIT, "h19-freeze/hala-h21-frozen.zip", H19_SHA, 3 * 1024**2),
                root / "frozen")
    adapter = root / "retained-adapter"
    adapter.mkdir()
    for name, expected in ADAPTER_FILES.items():
        bound = 4 * 1024**2 if name.endswith("safetensors") else 65536
        (adapter / name).write_bytes(public_artifact(ARTIFACT_COMMIT,
            "h21-shared-pilot/trained-adapter/" + name, expected, bound))
    sys.path.insert(0, str(source))
    from development.freeze import verify_bundle
    from training.adapter_archive import verify_bundle as verify_adapter
    verify_bundle(root / "frozen", require_media=True, expected_manifest_sha256=H19_MANIFEST)
    verify_adapter(adapter, ADAPTER_IDENTITY, ADAPTER_MANIFEST)
    # Immutable external evidence is verified before rerunning old TEST cases.
    for n, (commit, expected) in enumerate(DEV):
        mode = MODES[n]
        raw = public_artifact(commit, f"h22-attempt2/profile-{n}-{mode}-dev-evidence.zip",
                              expected, 16 * 1024**2)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if digest(z.read("experiment.json")) != DEV_SHA:
                raise ValueError("Retained DEV experiment hash mismatch")
    previous = public_artifact(EARLIER_FP16_COMMIT,
        "h22-test-restart/profile-0-fp16-test-evidence.zip", EARLIER_FP16_ARCHIVE, 16 * 1024**2)
    with zipfile.ZipFile(io.BytesIO(previous)) as z:
        last = json.loads(z.read("profile-status.json"))
        if last.get("index") != 0 or last.get("status") != "finished":
            raise ValueError("Earlier TEST evidence is not a completed FP16 run")
    return source


def qualify(source, output):
    os.environ["HALA_REMOTE_DEVELOPMENT_ALLOWED"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"):
        if key in os.environ and Path(os.environ[key]).resolve() != Path(CACHE).resolve():
            raise ValueError("Unreviewed cache environment override")
        os.environ[key] = CACHE
    from evaluation import development_precision as precision
    from evaluation.development_runtime import evaluation_guard
    from development.media import FONT_SHA256
    from PIL import features
    evaluation_guard("kaggle")
    precision.verify_sources(FREEZE)
    precision.check_cache(CACHE)
    for package, expected in {"torch": "2.11.0+cu128", "transformers": "5.16.1",
             "Pillow": "11.3.0", "accelerate": "1.12.0", "bitsandbytes": "0.50.2",
             "safetensors": "0.8.0", "peft": "0.20.0"}.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError("Unqualified environment package: " + package)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
        raise RuntimeError("Actual free Tesla T4 required")
    uuid = precision.resolve_device_uuid(torch)
    if features.version_feature("raqm") != "0.10.1":
        raise RuntimeError("RAQM rendering mismatch")
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if digest(font.read_bytes()) != FONT_SHA256:
        raise RuntimeError("Font content mismatch")
    report = {"schema": "hala.h22.batch_assigned_hardware.v1",
              "provider": "kaggle", "kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
              "gpu_name": torch.cuda.get_device_name(0), "gpu_uuid": uuid,
              "source_commit": SOURCE, "source_freeze_sha256": FREEZE,
              "h19_manifest_sha256": H19_MANIFEST, "adapter_manifest_sha256": ADAPTER_MANIFEST,
              "earlier_fp16_replicated_not_overwritten": True,
              "earlier_int8_unknown": True, "earlier_nf4_not_started": True,
              "batch_profile_order": list(MODES),
              "no_adoption": True, "training_runs": 0}
    atomic_json(output / "assigned-hardware.json", report)
    return report



def completion_counts(result):
    """Independent schema guard for measured eight-episode bilingual H19 result."""
    if type(result) is not dict or result.get("status") != "measured":
        raise ValueError("Only fully measured results have completion counts")
    summary = result.get("summary")
    if type(summary) is not dict or not {"all", "en", "ar"}.issubset(summary):
        raise ValueError("H19 language summary missing")
    completed = {}
    for language, total in (("all", 8), ("en", 4), ("ar", 4)):
        section = summary[language]
        if type(section) is not dict or type(section.get("episodes")) is not int or section["episodes"] != total:
            raise ValueError("Unexpected H19 episode count")
        value = section.get("episodes_complete")
        if type(value) is not int or not 0 <= value <= total:
            raise ValueError("Invalid independently verified completion count")
        completed[language] = value
    if completed["all"] != completed["en"] + completed["ar"]:
        raise ValueError("Per-language verified completion counts disagree")
    return completed


def run(root=ROOT):
    source = prepare(root)
    results = root / "results"
    results.mkdir()
    batch = {"schema": "hala.h22.batch_test_triplet.v1",
             "status": "preparing", "source_commit": SOURCE,
             "earlier_fp16_archive_sha256": EARLIER_FP16_ARCHIVE,
             "earlier_int8_outcome": "unknown", "earlier_nf4_outcome": "not_started",
             "profiles": [], "training_runs": 0, "model_adopted": False}
    atomic_json(results / "batch-status.json", batch)
    assigned = qualify(source, results)
    batch["assigned_hardware"] = assigned
    batch["status"] = "running"
    atomic_json(results / "batch-status.json", batch)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for idx, mode in enumerate(MODES):
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                   cwd=source, text=True).strip():
            raise RuntimeError("Scientific source changed during batch")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != SOURCE:
            raise RuntimeError("Scientific source commit changed during batch")
        # Observe the physical UUID again before every new process allocation.
        import torch
        from evaluation.development_precision import resolve_device_uuid
        if resolve_device_uuid(torch) != assigned["gpu_uuid"]:
            raise RuntimeError("Physical GPU changed during TEST batch")
        name = f"profile-{idx}-{mode}-test"
        folder = results / name
        folder.mkdir()
        args = [sys.executable, "-u", "-m", "evaluation.development_precision",
                "--mode", mode, "--source-freeze-sha256", FREEZE,
                "--bundle", str(root / "frozen"),
                "--manifest-sha256", H19_MANIFEST, "--output", str(folder / "run"),
                "--font-path", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "--source-commit", SOURCE, "--provider", "kaggle", "--split", "test",
                "--adapter", str(root / "retained-adapter"),
                "--adapter-manifest-sha256", ADAPTER_MANIFEST,
                "--adapter-identity", json.dumps(ADAPTER_IDENTITY, sort_keys=True)]
        record = {"mode": mode, "split": "test", "status": "running",
                  "started_unix": time.time(), "gpu_uuid": assigned["gpu_uuid"]}
        batch["profiles"].append(record)
        atomic_json(results / "batch-status.json", batch)
        try:
            with (folder / "profile.log").open("wb") as log:
                process = subprocess.run(args, cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=3700)
            record["status"] = "finished" if process.returncode == 0 else "failed"
            record["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
        except Exception as error:
            record["status"] = "launcher_error"
            record["error_type"] = type(error).__name__
        result = folder / "run" / "result.json"
        if result.is_file():
            try:
                content = result.read_bytes()
                record["result_sha256"] = digest(content)
                measured = json.loads(content)
                record["result_status"] = measured.get("status")
                record["completed_by_language"] = completion_counts(measured)
                if measured.get("gpu_uuid") != assigned["gpu_uuid"]:
                    record["status"] = "physical_device_mismatch"
            except (OSError, ValueError, TypeError, KeyError) as error:
                # Malformed/incomplete GPU output must NOT suppress archival of
                # the raw file, stdout, exit status, and previous profiles.
                record["status"] = "invalid_result"
                record["result_record_error"] = type(error).__name__
        record["finished_unix"] = time.time()
        atomic_json(folder / "profile-status.json", record)
        archive = results / (name + "-evidence.zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    if path.is_symlink() or path.stat().st_size > 8 * 1024**2:
                        raise ValueError("Evidence file exceeds bound; preserve raw directory")
                    z.write(path, path.relative_to(folder).as_posix())
            z.write(results / "assigned-hardware.json", "assigned-hardware.json")
            z.write(Path(__file__), "h22_batch_operator.py")
        if archive.stat().st_size > 16 * 1024**2:
            raise ValueError("Evidence ZIP exceeds limit; raw directory preserved")
        record["archive_sha256"] = digest(archive.read_bytes())
        record["archive_bytes"] = archive.stat().st_size
        atomic_json(results / "batch-status.json", batch)
        print("HALA_BATCH_PROFILE:", json.dumps(record, sort_keys=True), flush=True)
    batch["status"] = ("completed_unqualified" if all(
        x["status"] == "finished" and x.get("result_status") == "measured"
        for x in batch["profiles"]) else "completed_with_blockers")
    atomic_json(results / "batch-status.json", batch)
    print("HALA_BATCH_FINISHED:", json.dumps(batch, sort_keys=True), flush=True)
    return batch


if __name__ == "__main__":
    run()
