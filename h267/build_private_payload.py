"""Create PRIVATE H267 GPU code-only Kaggle script; zero datasets/weights.

All source is in the Hala GitHub repo, local payload is disposable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from h265.build_private_kaggle_payload import INSTALL

KERNEL_ID = "abdulazizshehri/hala-h267-private-real-vlm-json-diagnostic"
SOURCE_FILE = "free_kaggle_format_probe.py"
SCRIPT_FILE = "h267_private_kaggle.py"
META = "kernel-metadata.json"
INSTALL_H267 = INSTALL.replace(
    "HALA_H265_NONCOMMERCIAL_FREE_GPU",
    "HALA_H267_NONCOMMERCIAL_FREE_GPU",
)


def make(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE_FILE
            or not source.parent.name == "h267"
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or any(part.name == ".git" for part in destination.parents)):
        raise ValueError("H267 requires checked-in source and new scratch folder")
    text=source.read_text(encoding="utf-8")
    future="from __future__ import annotations\n"
    if text.count(future)!=1 or text.count("if __name__==\"__main__\":")!=1:
        raise ValueError("H267 source marker drift")
    payload=future+INSTALL_H267+"\n"+text.replace(future,"",1)
    meta={
        "id":KERNEL_ID,
        "title":"Hala H267 Private Real VLM JSON Diagnostic",
        "code_file":SCRIPT_FILE,
        "language":"python",
        "kernel_type":"script",
        "is_private":"true",
        "enable_gpu":"true",
        "enable_tpu":"false",
        "enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],
        "competition_sources":[],
        "kernel_sources":[],
        "model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_FILE).write_text(payload,encoding="utf-8")
    (destination/META).write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",
                                  encoding="utf-8")
    return {"kernel":KERNEL_ID,"private":True,"model_or_dataset_uploads":0,
            "free_gpu_type":"NvidiaTeslaT4","payload_python_files":1}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(make(args.source,args.destination),sort_keys=True))
