"""Opt-in REAL pinned Qwen3-VL processor-only parity on authored public H357/H358.

CPU-only; no model weights, optimizer, training admission, private media or blind OOD.
All source screenshots/labels are generated in a disposable temporary directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h216.real_qwen_processor_alignment import verify_cached_processor_root
from h357.source_candidate import capture_source_candidate
from h358.source_candidate import capture_candidate
from h361.original_qwen_processor import run_pinned_offline_candidate_processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-processor-snapshot", required=True, type=Path)
    parser.add_argument("--browser", choices=("msedge", "chromium"), required=True)
    args = parser.parse_args()
    if (ADMITTED_TRAIN_MANIFEST_SHAS
            or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
            or os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1"):
        raise SystemExit("H361_CPU_PUBLIC_PROCESSOR_PREFLIGHT_BLOCKED")
    # This script must not load or activate GPU even if a CUDA runtime is installed.
    import torch
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise SystemExit("H361_CPU_ONLY_REQUIRED")
    verify_cached_processor_root(args.local_processor_snapshot)
    with TemporaryDirectory(prefix="hala-h361-original-public-") as temporary:
        a = Path(temporary) / "controlgarden"
        b = Path(temporary) / "noterelay"
        capture_source_candidate(a, browser_kind=args.browser)
        capture_candidate(b, browser_kind=args.browser)
        receipt = run_pinned_offline_candidate_processor(
            args.local_processor_snapshot,
            {"controlgarden-single-app": a,
             "noterelay-original-multistep": b},
        )
    # Source images and original labels have been removed before any output.
    print("H361_REAL_CPU_PROCESSOR_ONLY",
          json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
