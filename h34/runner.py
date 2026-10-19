"""Owned-process bounded private Kaggle image inference; no action execution."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .corpus import build_cases, canonical_cases, case_sha256
from .render import fetch_pinned_font, load_font_pin, render_case

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PINS = {"torch": "2.11.0+cu128", "transformers": "5.16.1", "accelerate": "1.14.0",
        "safetensors": "0.8.0", "Pillow": "12.3.0"}
PROMPT = ("Identify the button whose visible label best matches the user's task. "
          "Return exactly one JSON object with normalized center coordinates: "
          '{"point":[x,y]} where each number is from 0 to 1. If no single button can be identified, '
          'return exactly {"abstain":true}. Do not output labels, explanations, or other fields.')


def admit():
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("H34 actual inference requires Kaggle Linux")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "Batch":
        raise RuntimeError("H34 requires a retained Kaggle Batch run")
    if os.environ.get("HALA_H34_RESEARCH_ALLOWED") != "1":
        raise RuntimeError("explicit H34 private research opt-in required")


def configure_environment():
    # Set before importing torch, transformers or any Hub client.
    os.environ.update(HF_HOME="/tmp/hala-h34-cache", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", DO_NOT_TRACK="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(source_commit):
    root = Path(__file__).resolve().parent.parent
    if not re.fullmatch("[a-f0-9]{40}", source_commit):
        raise ValueError("exact source commit required")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, timeout=20)
    if git("rev-parse", "HEAD").decode().strip() != source_commit or git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("clean exact source commit required")
    from tools.build_h34_kaggle import FILES
    files = {}
    for name in FILES:
        payload = (root / name).read_bytes()
        if payload != git("show", f"{source_commit}:{name}"):
            raise ValueError("source bytes differ from immutable commit")
        files[name] = hashlib.sha256(payload).hexdigest()
    if (root / "h34/frozen/cases.json").read_bytes() != canonical_cases():
        raise ValueError("frozen corpus differs from evaluator")
    pin = load_font_pin()
    if hashlib.sha256((root / pin["license_file"]).read_bytes()).hexdigest() != pin["license_sha256"]:
        raise ValueError("frozen font license differs")
    return files


def _load_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H34 explicitly requires a T4 GPU")
    options = dict(revision=REVISION, trust_remote_code=False, token=False,
                   cache_dir="/tmp/hala-h34-cache")
    processor = AutoProcessor.from_pretrained(MODEL, **options)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, **options, use_safetensors=True, dtype=torch.float16,
        device_map="cuda:0").eval()
    return processor, model


def owned_process(command, timeout_seconds, *, cwd=None):
    """Kill and reap this fresh process group, including descendants, on deadline/exit."""
    if platform.system() != "Linux":
        raise RuntimeError("owned process groups require Linux")
    # Adopt and reap worker descendants instead of leaving zombies to notebook PID 1.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Linux child-subreaper setup failed")
    proc = None
    try:
        proc = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            return proc.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            return None, True
    finally:
        try:
            if proc is not None:
                # Even successful children must not leave descendants behind.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                reap_deadline = time.monotonic() + 10
                while True:
                    try:
                        child, _ = os.waitpid(-proc.pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if child == 0:
                        if time.monotonic() >= reap_deadline:
                            raise RuntimeError("owned descendants did not reap")
                        time.sleep(0.01)
        finally:
            if libc.prctl(36, previous.value, 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Linux child-subreaper restore failed")


def _worker(output, source_commit):
    configure_environment()
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_path = output / "raw_attempts.jsonl"
    try:
        metadata["source_files"] = source_identity(source_commit)
        cases = build_cases()
        if len(cases) != 32:
            raise ValueError("frozen case count changed")
        (output / "cases.sha256").write_bytes((case_sha256() + "\n").encode("ascii"))
        pin = load_font_pin()
        metadata["font"] = pin
        try:
            hardware = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
                text=True, timeout=10)
            metadata["gpu_observations"] = [dict(zip(("index", "name", "uuid", "vram_mib"),
                                                     (v.strip() for v in row)))
                                            for row in csv.reader(hardware.splitlines())]
        except (OSError, subprocess.SubprocessError) as exc:
            metadata["gpu_observation_failure"] = type(exc).__name__
        metadata["packages"] = {}
        for name in PINS:
            try:
                metadata["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                metadata["packages"][name] = None
        save_json(metadata_path, metadata)
        if metadata["packages"] != PINS:
            raise RuntimeError("unqualified package versions")
        import torch
        metadata["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                             "uuid": str(torch.cuda.get_device_properties(i).uuid),
                             "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())]
        save_json(metadata_path, metadata)
        from PIL import Image, features
        metadata["raqm"] = features.version_feature("raqm")
        save_json(metadata_path, metadata)
        if metadata["raqm"] != "0.10.5":
            raise RuntimeError("unqualified RAQM version")
        font = fetch_pinned_font(Path("/tmp/hala-h34-cache/DejaVuSans.ttf"), pin)
        images = []
        save_json(output / "image_manifest.json", images)
        for case in cases:
            media = render_case(case, output / (case["id"] + ".png"), font, pin)
            if media["bytes"] > 524288:
                raise RuntimeError("rendered image exceeded byte bound")
            media["case_id"] = case["id"]
            images.append(media)
            save_json(output / "image_manifest.json", images)
        metadata["state"] = "loading_model"
        save_json(metadata_path, metadata)
        processor, model = _load_model()
        metadata["state"] = "inference_started"
        save_json(metadata_path, metadata)
        with raw_path.open("a", encoding="utf-8", newline="\n") as journal:
            for index, case in enumerate(cases):
                record = {"case_id": case["id"], "language": case["language"], "raw": None,
                          "error": "inference_interrupted", "image_sha256": images[index]["sha256"]}
                save_json(output / "pending_attempt.json", record)
                try:
                    with (output / (case["id"] + ".png")).open("rb") as image_file:
                        pixels = image_file.read(524289)
                    if len(pixels) > 524288 or hashlib.sha256(pixels).hexdigest() != record["image_sha256"]:
                        raise ValueError("image bytes changed after rendering")
                    with Image.open(io.BytesIO(pixels)) as png:
                        image = png.convert("RGB")
                    content = [{"type": "image", "image": image},
                               {"type": "text", "text": case["goal"] + "\n" + PROMPT}]
                    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                         tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=96)
                    raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
                        raise ValueError("oversized raw response")
                    record.update(raw=raw, error=None)
                except Exception as exc:
                    record.update(raw=None, error=type(exc).__name__)
                journal.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                (output / "pending_attempt.json").unlink()
                print("H34_CASE", index + 1, case["id"], "attempt_saved", flush=True)
        metadata["state"] = "completed"
    except Exception as exc:
        metadata.update(state="incomplete", failure_type=type(exc).__name__)
    save_json(metadata_path, metadata)


def run(output: Path, source_commit: str, timeout_seconds: int = 1200):
    admit()
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1200:
        raise ValueError("deadline must be 1..1200 seconds")
    configure_environment()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "run_metadata.json"
    metadata = {"state": "starting", "source_commit": source_commit, "case_sha256": case_sha256(),
                "model": MODEL, "revision": REVISION, "precision": "float16", "token": False,
                "trust_remote_code": False, "use_safetensors": True, "expected_cases": 32,
                "split": "prospective-synthetic-dev", "actions_executed": False,
                "cache": "/tmp/hala-h34-cache", "deadline_seconds": timeout_seconds,
                "observations_are_attestation": False, "resume": False}
    save_json(metadata_path, metadata)
    raw_path = output / "raw_attempts.jsonl"
    raw_path.touch(exist_ok=False)
    started = time.monotonic()
    code, timed_out, supervisor_failure = None, False, None
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("terminated")))
    try:
        code, timed_out = owned_process([sys.executable, "-u", "-m", "h34.runner", "--worker",
                                        "--output", str(output), "--source-commit", source_commit], timeout_seconds)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        supervisor_failure = type(exc).__name__
    finally:
        signal.signal(signal.SIGTERM, previous)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if supervisor_failure:
        metadata["supervisor_failure"] = supervisor_failure
    records = raw_path.read_bytes().splitlines()
    pending = output / "pending_attempt.json"
    if pending.exists():
        record = json.loads(pending.read_text(encoding="utf-8"))
        # A kill between fsync and pending-file removal must not duplicate the row.
        if len(records) < 32 and (not records or json.loads(records[-1])["case_id"] != record["case_id"]):
            with raw_path.open("ab") as journal:
                journal.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                journal.flush()
                os.fsync(journal.fileno())
    count = len(raw_path.read_bytes().splitlines())
    if timed_out or code != 0 or metadata["state"] != "completed" or count != 32:
        metadata["state"] = "incomplete"
    metadata.update(deadline_exceeded=timed_out, worker_exit_code=code,
                    attempted_count=count, elapsed_seconds=time.monotonic()-started,
                    raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    save_json(metadata_path, metadata)
    print("H34_EXTERNAL_RAW_SHA256", metadata["raw_sha256"], flush=True)
    print("H34_RAW_AUDIT_REQUIRED; MODEL_QUALIFICATION_NOT_ESTABLISHED", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        admit()
        _worker(Path(args.output), args.source_commit)
    else:
        run(Path(args.output), args.source_commit)


if __name__ == "__main__":
    main()
