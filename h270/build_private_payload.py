"""Create-only owner-private Kaggle payload for H270 first-party weighted LoRA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from h269.build_private_payload import INSTALL_H269

KERNEL_ID="abdulazizshehri/hala-h270-private-coordinate-weighted-lora-t4"
SCRIPT_NAME="h270_private_coordinate_weighted.py"
META_NAME="kernel-metadata.json"
SOURCE_NAME="coordinate_weighted_lora.py"
INSTALL_H270=INSTALL_H269.replace(
    "HALA_H269_PRIVATE_FREE_NONCOMMERCIAL",
    "HALA_H270_PRIVATE_NONCOMMERCIAL_FREE_T4",
)


def make(source:Path,destination:Path)->dict:
    if (not isinstance(source,Path) or not source.is_file()
            or source.parent.name!="h270" or source.name!=SOURCE_NAME
            or not isinstance(destination,Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H270 only checked-in source and new external scratch")
    body=source.read_text("utf-8")
    future="from __future__ import annotations\n"
    if body.count(future)!=1 or 'if __name__=="__main__":' not in body:
        raise ValueError("H270 source identity marker drift")
    data=future+INSTALL_H270+"\n"+body.replace(future,"",1)
    meta={
        "id":KERNEL_ID,
        "title":"Hala H270 Private Coordinate Weighted LoRA T4",
        "code_file":SCRIPT_NAME,"language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_tpu":"false",
        "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],"competition_sources":[],
        "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_NAME).write_text(data,encoding="utf-8")
    (destination/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"private":True,"kernel":KERNEL_ID,"gpu":"free NvidiaTeslaT4",
            "model_or_media_uploads":0,"script_files":1}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--destination",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(make(args.source,args.destination),sort_keys=True))
