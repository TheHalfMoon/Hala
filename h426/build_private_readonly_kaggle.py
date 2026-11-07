"""Deterministically package PUBLIC source code for a PRIVATE Kaggle read-only run.

The bundle contains source .py only, never weights, private data, prompts,
model predictions, secrets, credentials, images, or original training labels.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARTS = (
    "h218/pinned_qwen_processor_bytes.py",
    "h387/verified_processor_handoff.py",
    "h425/kaggle/h425_h417_replay_paired.py",
    "h425/kaggle/h425_h421_paired_eval.py",
    "h426/h425_loss_bridge.py",
    "h426/synthetic_drag_curriculum.py",
    "h426/synthetic_h425_processor_bridge.py",
    "h426/synthetic_pair_eval.py",
    "h426/synthetic_inference_eval.py",
    "h426/paired_study_statistics.py",
    "h426/verify_private_h425_baseline.py",
    "h426/kaggle_h425_readonly_synthetic.py",
)
KERNEL_ID = "abdulazizshehri/hala-h426-private-h425-synthetic-readonly"
PARENT_DATASET = "abdulazizshehri/hala-h425-genuine-trained-parent-private"
MAX_BUNDLE_BYTES = 2 * 1024 * 1024


def source_bundle(root: Path = ROOT) -> bytes:
    if not root.is_dir():
        raise ValueError("H426 source root unavailable")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for filename in PARTS:
            path = root / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("H426 frozen source missing or linked: "+filename)
            data = path.read_bytes()
            if (not data or len(data) > MAX_BUNDLE_BYTES
                or b"\x00" in data):
                raise ValueError("H426 source is not bounded plain Python")
            info = zipfile.ZipInfo(filename, date_time=(2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    blob = output.getvalue()
    if not 0 < len(blob) < MAX_BUNDLE_BYTES:
        raise ValueError("H426 source bundle too large")
    return blob


def script(bundle: bytes) -> str:
    digest = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode("ascii")
    chunks = "\n".join("    "+repr(encoded[i:i+100])
                       for i in range(0,len(encoded),100))
    return (
        '"""Owner-private H426 source-only Kaggle inference bundle."""\n'
        "from __future__ import annotations\n"
        "import base64, hashlib, io, os, pathlib, runpy, sys, tempfile, zipfile\n"
        "PAYLOAD = (\n"+chunks+"\n)\n"
        "payload = base64.b64decode(PAYLOAD, validate=True)\n"
        f"if hashlib.sha256(payload).hexdigest() != '{digest}':\n"
        '    raise ValueError("H426 private Kaggle source bundle SHA drift")\n'
        "with tempfile.TemporaryDirectory(prefix='.h426-src-',dir='/kaggle/working') as tmp:\n"
        "    root = pathlib.Path(tmp)\n"
        "    with zipfile.ZipFile(io.BytesIO(payload)) as z:\n"
        f"        if len(z.infolist()) != {len(PARTS)}:\n"
        '            raise ValueError("H426 unexpected source count")\n'
        "        for entry in z.infolist():\n"
        "            rel = pathlib.PurePosixPath(entry.filename)\n"
        "            if rel.is_absolute() or '..' in rel.parts or entry.is_dir():\n"
        '                raise ValueError("H426 unsafe private bundle pathname")\n'
        "            if not entry.filename.endswith('.py') or entry.file_size > 2*1024*1024:\n"
        '                raise ValueError("H426 invalid source in bundle")\n'
        "            target = root.joinpath(*rel.parts)\n"
        "            target.parent.mkdir(parents=True, exist_ok=True)\n"
        "            target.write_bytes(z.read(entry))\n"
        "    sys.path.insert(0,str(root))\n"
        "    runpy.run_path(str(root/'h426/kaggle_h425_readonly_synthetic.py'),run_name='__main__')\n"
    )


def metadata():
    return {
        "id": KERNEL_ID,
        "title": "Hala H426 Private H425 Synthetic Readonly",
        "code_file": "h426_private_h425_synthetic_readonly.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # Public official Qwen weights/processor only. H425 LoRA private input.
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [PARENT_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build(destination: Path, root: Path = ROOT):
    if destination.resolve() == root.resolve() or root.resolve() in destination.resolve().parents:
        raise ValueError("H426 private Kaggle work products must stay out of repository")
    destination.mkdir(parents=True,exist_ok=True)
    payload = source_bundle(root)
    source_file = destination / metadata()["code_file"]
    metadata_file = destination / "kernel-metadata.json"
    if source_file.exists() or metadata_file.exists():
        raise ValueError("H426 private Kaggle launch artifacts are create-only")
    source_file.write_text(script(payload),encoding="utf8")
    metadata_file.write_text(json.dumps(metadata(),sort_keys=True,indent=2)+"\n",
                             encoding="utf8")
    return {
        "status":"READY_FOR_OWNER_PRIVATE_KAGGLE_READONLY_SOURCE_ONLY",
        "kernel_id": KERNEL_ID,
        "parts":len(PARTS),
        "source_bundle_sha256":hashlib.sha256(payload).hexdigest(),
        "private_script_sha256":hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "is_private":True,
        "trainable_model":False,
        "kaggle_run_started":False,
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-directory",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.output_directory),sort_keys=True))
