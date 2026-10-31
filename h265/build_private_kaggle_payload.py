"""Build a PRIVATE, free-T4 Kaggle kernel from checked-in H265 source.

All reproducible code lives in public Hala Git. The local payload directory
is TEMPORARY, not a data source; delete it after private kernel submission.
No owner-private media, gold or credentials are packaged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

KERNEL_ID = "abdulazizshehri/hala-h265-private-qwen3vl-firstparty-t4"
TITLE = "Hala H265 Private Qwen3-VL First-Party T4"
SCRIPT_NAME = "h265_private_kaggle.py"
META_NAME = "kernel-metadata.json"
INSTALL = """# Kaggle entry: start only on private, opted-in noncommercial FREE T4
import os
import sys
import subprocess
if sys.platform != 'linux' or not os.path.isdir('/kaggle/working'):
    raise RuntimeError('H265 Kaggle Linux only')
if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    raise RuntimeError('H265 must execute inside a real Kaggle kernel')
import torch
if not torch.cuda.is_available() or not all(
    'T4' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
):
    raise RuntimeError('H265 private FREE NvidiaTeslaT4 session required')
os.environ['HALA_H265_NONCOMMERCIAL_FREE_GPU'] = '1'
# Torch is NEVER replaced; only bounded no-cost Python inference libraries.
subprocess.run([sys.executable, '-m', 'pip', 'install',
    '--disable-pip-version-check', '-q',
    'transformers==5.16.1', 'playwright==1.58.0'], check=True)
subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium'], check=True)
"""

def build_payload(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or any(parent.name == '.git' for parent in destination.parents)):
        raise ValueError("H265 create-only scratch folder outside source")
    text=source.read_text(encoding='utf-8')
    target = "from __future__ import annotations\n"
    if text.count(target)!=1 or "if __name__ == \"__main__\":" not in text:
        raise ValueError("H265 original source contract changed")
    payload = target + INSTALL + "\n" + text.replace(target,"",1)
    metadata = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT_NAME,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources":[], "competition_sources":[],
        "kernel_sources":[], "model_sources":[]
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination / SCRIPT_NAME).write_text(payload, encoding='utf-8')
    (destination / META_NAME).write_text(
        json.dumps(metadata,sort_keys=True,indent=2)+"\n", encoding='utf-8')
    return {
        "kernel_id": KERNEL_ID,
        "private": metadata["is_private"] == "true",
        "requested_free_t4":metadata["machine_shape"],
        "source_files":1, "original_data_uploaded":False,
        "model_weights_uploaded":False, "credentials_uploaded":False,
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_payload(args.source,args.destination),sort_keys=True))
