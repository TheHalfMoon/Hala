"""H36 paired within-VM synthetic DEV inference; no OS action execution.

Reuses the reviewed H34 Kaggle T4/FP16 loader and process-group ownership.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from h34.corpus import build_cases, case_sha256
from h34.render import fetch_pinned_font, load_font_pin, render_case
from h34.runner import (MODEL, PINS, REVISION, _load_model, configure_environment,
                         owned_process, save_json)
from .protocol import (ARMS, PROTOCOL, instructions, plan, plan_sha,
                       prompt_sha, sha_bytes)


def admit() -> None:
    # Keep Linux/Batch constraints independent of H34-specific opt-in.
    import platform
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("H36 requires Kaggle Linux")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "Batch":
        raise RuntimeError("H36 requires retained Kaggle Batch")
    if os.environ.get("HALA_H36_RESEARCH_ALLOWED") != "1":
        raise RuntimeError("explicit private H36 noncommercial research opt-in required")


def source_identity(source_commit: str) -> dict:
    if not re.fullmatch("[a-f0-9]{40}", source_commit):
        raise ValueError("immutable H36 source commit required")
    root = Path(__file__).resolve().parent.parent
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root, timeout=25)
    if git("rev-parse", "HEAD").decode().strip() != source_commit:
        raise ValueError("git HEAD differs from required source")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("source must be clean and immutable")
    from tools.build_h36_kaggle import FILES
    result = {}
    for name in FILES:
        blob = (root / name).read_bytes()
        if blob != git("show", f"{source_commit}:{name}"):
            raise ValueError("H36 source bytes differ from signed commit")
        result[name] = sha_bytes(blob)
    return result


def _worker(output: Path, source_commit: str) -> None:
    configure_environment()
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_path = output / "raw_attempts.jsonl"
    try:
        metadata["source_files"] = source_identity(source_commit)
        cases = build_cases()
        tasks = plan()
        (output / "cases.sha256").write_text(case_sha256() + "\n", encoding="ascii")
        (output / "plan.sha256").write_text(plan_sha() + "\n", encoding="ascii")
        font_pin = load_font_pin()
        if sha_bytes((Path(__file__).parent.parent / font_pin["license_file"]).read_bytes()) != font_pin["license_sha256"]:
            raise ValueError("frozen font license SHA mismatch")
        metadata["font"] = font_pin
        metadata["packages"] = {}
        for name in PINS:
            try:
                metadata["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                metadata["packages"][name] = None
        try:
            txt = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10)
            metadata["gpu_observations"] = [list(map(str.strip, line))
                                            for line in csv.reader(txt.splitlines())]
        except (OSError, subprocess.SubprocessError) as exc:
            metadata["gpu_observation_failure"] = type(exc).__name__
        save_json(metadata_path, metadata)
        if metadata["packages"] != PINS:
            raise RuntimeError("H36 requires same H34 pinned package versions")
        from PIL import Image, features
        if features.version_feature("raqm") != "0.10.5":
            raise RuntimeError("H36 requires same pinned RAQM")
        font = fetch_pinned_font(Path("/tmp/hala-h34-cache/DejaVuSans.ttf"), font_pin)
        images = []
        save_json(output / "image_manifest.json", images)
        for case in cases:
            record = render_case(case, output / (case["id"] + ".png"), font, font_pin)
            if record["bytes"] > 524288:
                raise RuntimeError("PNG exceeded hard byte bound")
            record["case_id"] = case["id"]
            images.append(record)
            save_json(output / "image_manifest.json", images)
        metadata["state"] = "loading_model"
        save_json(metadata_path, metadata)
        processor, model = _load_model()
        import torch
        metadata["state"] = "paired_inference"
        metadata["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                             "uuid": str(torch.cuda.get_device_properties(i).uuid),
                             "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())]
        save_json(metadata_path, metadata)
        lookup = {case["id"]: case for case in cases}
        pngs = {img["case_id"]: img for img in images}
        with raw_path.open("a", encoding="utf-8", newline="\n") as journal:
            for index, trial in enumerate(tasks):
                case = lookup[trial["case_id"]]
                ref = pngs[case["id"]]
                record = {"case_id": case["id"], "language": case["language"],
                          "arm": trial["arm"], "image_sha256": ref["sha256"],
                          "prompt_sha256": prompt_sha(trial["arm"]),
                          "raw": None, "error": "inference_interrupted"}
                save_json(output / "pending_attempt.json", record)
                try:
                    path = output / (case["id"] + ".png")
                    pixels = path.read_bytes()
                    if len(pixels) > 524288 or sha_bytes(pixels) != ref["sha256"]:
                        raise ValueError("image bytes changed after rendering")
                    with Image.open(io.BytesIO(pixels)) as image_file:
                        image = image_file.convert("RGB")
                    content = [{"type": "image", "image": image},
                               {"type": "text", "text": instructions(case, trial["arm"])}]
                    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                         tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        tokens = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=96)
                    raw = processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:],
                                                 skip_special_tokens=True)[0]
                    if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
                        raise ValueError("oversized model response")
                    record.update(raw=raw, error=None)
                except Exception as exc:
                    record.update(raw=None, error=type(exc).__name__)
                journal.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                (output / "pending_attempt.json").unlink()
                print("H36_PAIR", index + 1, case["id"], trial["arm"], "retained", flush=True)
        metadata["state"] = "completed"
    except Exception as exc:
        metadata.update(state="incomplete", failure_type=type(exc).__name__)
    save_json(metadata_path, metadata)


def run(output: Path, source_commit: str, timeout_seconds: int = 1200) -> dict:
    admit()
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1200:
        raise ValueError("bounded deadline required")
    configure_environment()
    output = Path(output).resolve()
    if not output.is_relative_to(Path("/kaggle/working")):
        raise ValueError("private Kaggle output must be in /kaggle/working")
    output.mkdir(parents=True, exist_ok=False)
    meta = {"protocol": PROTOCOL, "plan_sha256": plan_sha(),
            "source_commit": source_commit, "model": MODEL, "revision": REVISION,
            "split": "exposed-synthetic-dev", "expected_trials": 64,
            "prompt_sha256": {arm: prompt_sha(arm) for arm in ARMS},
            "actions_executed": False, "model_qualified": False,
            "observations_are_attestation": False, "state": "starting",
            "deadline_seconds": timeout_seconds, "resume": False}
    meta_path = output / "run_metadata.json"
    save_json(meta_path, meta)
    raw_path = output / "raw_attempts.jsonl"
    raw_path.touch(exist_ok=False)
    start = time.monotonic()
    exit_code, timed_out, supervision_error = None, False, None
    try:
        exit_code, timed_out = owned_process([sys.executable, "-u", "-m", "h36.runner",
                                               "--worker", "--output", str(output),
                                               "--source-commit", source_commit],
                                              timeout_seconds)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        supervision_error = type(exc).__name__
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    pending = output / "pending_attempt.json"
    if pending.exists():
        record = json.loads(pending.read_text(encoding="utf-8"))
        previous = raw_path.read_bytes().splitlines()
        if len(previous) < 64:
            last = json.loads(previous[-1]) if previous else None
            if last is None or (last["case_id"], last["arm"]) != (record["case_id"], record["arm"]):
                with raw_path.open("ab") as journal:
                    journal.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode())
                    journal.flush()
                    os.fsync(journal.fileno())
    actual = len(raw_path.read_bytes().splitlines())
    if timed_out or exit_code != 0 or metadata.get("state") != "completed" or actual != 64:
        metadata["state"] = "incomplete"
    metadata.update(worker_exit_code=exit_code, supervisor_error=supervision_error,
                    deadline_exceeded=timed_out, attempted_count=actual,
                    elapsed_seconds=time.monotonic()-start,
                    raw_sha256=sha_bytes(raw_path.read_bytes()))
    save_json(meta_path, metadata)
    print("H36_EXTERNAL_RAW_SHA256", metadata["raw_sha256"], flush=True)
    print("H36_REQUIRED_INDEPENDENT_AUDIT; NO_MODEL_QUALIFICATION", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(Path(args.output), args.source_commit)
    else:
        run(Path(args.output), args.source_commit)


if __name__ == "__main__":
    main()
