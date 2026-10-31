"""One-file code-only private free-T4 Kaggle H279 packager."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from h273.build_private_payload import INSTALL_H273

KERNEL_ID="abdulazizshehri/hala-h279-private-relative-native-actions-t4"
SCRIPT_NAME="h279_private_relative_native.py"
META_NAME="kernel-metadata.json"
SOURCE_NAME="private_relative_native_probe.py"
INSTALL_H279=INSTALL_H273.replace(
    "HALA_H273_FREE_PRIVATE_NONCOMMERCIAL",
    "HALA_H279_PRIVATE_FREE_NONCOMMERCIAL")

def build(source:Path,destination:Path)->dict:
    if (not isinstance(source,Path) or not source.is_file()
        or source.parent.name!="h279" or source.name!=SOURCE_NAME
        or not isinstance(destination,Path) or destination.exists()
        or destination.is_relative_to(source.parent)
        or ".git" in destination.parts):
        raise ValueError("H279 must package checked-in single source only")
    text=source.read_text("utf-8")
    future="from __future__ import annotations\n"
    if (text.count(future)!=1
        or 'if __name__=="__main__":' not in text
        or "def classify_strict_model_text(" not in text
        or "raw=policy(png,RELATIVE_PROMPT)" not in text
        or "strict_native=normalize_strict_action(raw)" not in text
        or "outcome=execute_one(page,strict_native)" not in text):
        raise ValueError("H279 private diagnostic source contract drift")
    full_script=future+INSTALL_H279+"\n"+text.replace(future,"",1)
    meta={
        "id":KERNEL_ID,
        "title":"Hala H279 Private Relative Native Actions T4",
        "code_file":SCRIPT_NAME,"language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_tpu":"false",
        "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],"competition_sources":[],
        "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True,exist_ok=False)
    (destination/SCRIPT_NAME).write_text(full_script,encoding="utf-8")
    (destination/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"kernel":KERNEL_ID,"private":True,"free_gpu":"NvidiaTeslaT4",
            "uploaded_model_or_media":0,"python_scripts":1}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    opts=parser.parse_args()
    print(json.dumps(build(opts.source,opts.destination),sort_keys=True))
