"""H268 reproducible private code-only FREE T4 Kaggle packaging.

This creates a disposable 2-file upload folder OUTSIDE the public source
repository and never embeds media, model weights, tokens or credentials.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from h265.build_private_kaggle_payload import INSTALL

KERNEL_ID = "abdulazizshehri/hala-h268-private-constrained-native-t4"
SOURCE_NAME = "free_t4_constrained_native.py"
SCRIPT_NAME = "h268_private_t4.py"
META_NAME = "kernel-metadata.json"
INSTALL_H268 = INSTALL.replace(
    "HALA_H265_NONCOMMERCIAL_FREE_GPU",
    "HALA_H268_NONCOMMERCIAL_FREE_GPU",
) + """
# Token-level filter via lm-format-enforcer, locked to exactly one public
# MIT-licensed version, NEVER replaces Kaggle's CUDA PyTorch.
subprocess.run([sys.executable, '-m', 'pip', 'install',
    '--disable-pip-version-check', '-q',
    'git+https://github.com/noamgat/lm-format-enforcer.git@817f944fcc8851917c40300a47f62d1defc3ffc3'], check=True)
"""


def build_payload(source: Path, destination: Path) -> dict:
    if (not isinstance(source,Path) or not source.is_file()
            or source.name != SOURCE_NAME or source.parent.name != "h268"
            or not isinstance(destination,Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or any(part.name==".git" for part in destination.parents)):
        raise ValueError("H268 only fresh external scratch and pinned source")
    content=source.read_text(encoding="utf-8")
    future="from __future__ import annotations\n"
    if content.count(future)!=1 or "if __name__==\"__main__\":" not in content:
        raise ValueError("H268 source entrypoint marker drift")
    payload=future+INSTALL_H268+"\n"+content.replace(future,"",1)
    meta={
        "id":KERNEL_ID,
        "title":"Hala H268 Private Constrained Native T4",
        "code_file":SCRIPT_NAME,
        "language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true",
        "enable_tpu":"false","enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],"competition_sources":[],
        "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_NAME).write_text(payload,encoding="utf-8")
    (destination/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"private":True,"slug":KERNEL_ID,
            "free_gpu":"NvidiaTeslaT4","attached_data_or_weights":0,
            "python_files":1}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--destination",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(build_payload(a.source,a.destination),sort_keys=True))
