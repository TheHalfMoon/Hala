"""Build immutable owner-private H271 free T4 code-only Kaggle kernel."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from h270.build_private_payload import INSTALL_H270

KERNEL_ID="abdulazizshehri/hala-h271-private-vision-lora-native-t4"
SCRIPT_NAME="h271_private_vision_native.py"
META_NAME="kernel-metadata.json"
SOURCE_NAME="vision_lora_native.py"
INSTALL_H271=INSTALL_H270.replace(
    "HALA_H270_PRIVATE_NONCOMMERCIAL_FREE_T4",
    "HALA_H271_PRIVATE_NONCOMMERCIAL_FREE_T4",
)


def make(source:Path,destination:Path)->dict:
    if (not isinstance(source,Path) or not source.is_file()
            or source.parent.name!="h271" or source.name!=SOURCE_NAME
            or not isinstance(destination,Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H271 only checked-in source and create-only external scratch")
    body=source.read_text("utf-8")
    future="from __future__ import annotations\n"
    if body.count(future)!=1 or 'if __name__=="__main__":' not in body:
        raise ValueError("H271 private source markers drift")
    payload=future+INSTALL_H271+"\n"+body.replace(future,"",1)
    meta={
      "id":KERNEL_ID,"title":"Hala H271 Private Vision LoRA Native T4",
      "code_file":SCRIPT_NAME,"language":"python","kernel_type":"script",
      "is_private":"true","enable_gpu":"true","enable_tpu":"false",
      "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
      "dataset_sources":[],"competition_sources":[],
      "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_NAME).write_text(payload,encoding="utf-8")
    (destination/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"private":True,"kernel":KERNEL_ID,"free_gpu":"NvidiaTeslaT4",
            "weights_or_media_attached":0,"python_scripts":1}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(make(a.source,a.destination),sort_keys=True))
