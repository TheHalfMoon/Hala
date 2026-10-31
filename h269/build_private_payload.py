"""Build an owner-private free-Kaggle LoRA pilot from public source only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from h265.build_private_kaggle_payload import INSTALL

KERNEL_ID = "abdulazizshehri/hala-h269-private-firstparty-lora-t4"
SOURCE_NAME = "firstparty_lora_probe.py"
SCRIPT_NAME = "h269_private_firstparty_lora.py"
META_NAME = "kernel-metadata.json"
LMFE_GIT_PIN = "817f944fcc8851917c40300a47f62d1defc3ffc3"
INSTALL_H269 = INSTALL.replace(
    "HALA_H265_NONCOMMERCIAL_FREE_GPU",
    "HALA_H269_PRIVATE_FREE_NONCOMMERCIAL",
) + f"""
# Owner-authorized noncommercial FREE T4 only; no pretrained weights upload.
# Keep Kaggle CUDA PyTorch unchanged, verify before and after pip install.
from importlib.metadata import version as pkg_version
old_torch = pkg_version("torch")
subprocess.run([sys.executable, '-m','pip','install',
    '--disable-pip-version-check','-q',
    'peft==0.20.0',
    'git+https://github.com/noamgat/lm-format-enforcer.git@{LMFE_GIT_PIN}'],
    check=True)
if pkg_version("torch") != old_torch:
    raise RuntimeError("H269 refuses replacement of Kaggle CUDA PyTorch")
"""


def build_payload(source: Path, destination: Path) -> dict:
    if (not isinstance(source,Path) or not source.is_file()
            or source.parent.name!="h269" or source.name!=SOURCE_NAME
            or not isinstance(destination,Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H269 source must be checked-in, scratch create-only")
    text=source.read_text("utf-8")
    future="from __future__ import annotations\n"
    if text.count(future)!=1 or 'if __name__=="__main__":' not in text:
        raise ValueError("H269 source contract drift")
    script=future+INSTALL_H269+"\n"+text.replace(future,"",1)
    meta={
      "id":KERNEL_ID,"title":"Hala H269 Private Firstparty LoRA T4",
      "code_file":SCRIPT_NAME,"language":"python","kernel_type":"script",
      "is_private":"true","enable_gpu":"true","enable_tpu":"false",
      "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
      "dataset_sources":[],"competition_sources":[],
      "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_NAME).write_text(script,encoding="utf-8")
    (destination/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"kernel":KERNEL_ID,"private":True,"free_gpu":"NvidiaTeslaT4",
            "attached_data_or_weights":0,"payload_python_files":1}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--destination",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(build_payload(a.source,a.destination),sort_keys=True))
