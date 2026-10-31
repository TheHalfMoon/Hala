"""Build one code-only private H273 free-T4 kernel from two pinned sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from h265.build_private_kaggle_payload import INSTALL

KERNEL_ID="abdulazizshehri/hala-h273-private-relative-pixel-comparison-t4"
SCRIPT_NAME="h273_private_relative_pixel.py"
META_NAME="kernel-metadata.json"
LMFE_SHA="817f944fcc8851917c40300a47f62d1defc3ffc3"
INSTALL_H273=INSTALL.replace(
    "HALA_H265_NONCOMMERCIAL_FREE_GPU",
    "HALA_H273_FREE_PRIVATE_NONCOMMERCIAL",
)+f"""
from importlib.metadata import version as _version
_old_torch=_version("torch")
subprocess.run([sys.executable,"-m","pip","install",
    "--disable-pip-version-check","-q",
    "git+https://github.com/noamgat/lm-format-enforcer.git@{LMFE_SHA}"],
    check=True)
if _version("torch")!=_old_torch:
    raise RuntimeError("H273 cannot replace free Kaggle CUDA PyTorch")
"""


def make(source:Path,bridge:Path,dest:Path)->dict:
    if (not isinstance(source,Path) or not source.is_file()
        or source.parent.name!="h273" or source.name!="private_relative_probe.py"
        or not isinstance(bridge,Path) or not bridge.is_file()
        or bridge.parent!=source.parent or bridge.name!="relative_native_bridge.py"
        or not isinstance(dest,Path) or dest.exists()
        or dest.is_relative_to(source.parent) or ".git" in dest.parts):
        raise ValueError("H273 requires two checked-in firstparty text sources")
    body=source.read_text("utf-8")
    adapter=bridge.read_text("utf-8")
    future="from __future__ import annotations\n"
    if (body.count(future)!=1 or adapter.count(future)!=1
        or 'if __name__=="__main__":' not in body
        or "def normalize_strict_action(" not in adapter
        or "def build_normalized_token_filter(" not in adapter
        or "from h273.relative_native_bridge import" in body):
        raise ValueError("H273 fixed source bundling markers drift")
    script=future+INSTALL_H273+"\n"+adapter.replace(future,"",1)
    script+="\n"+body.replace(future,"",1)
    meta={
      "id":KERNEL_ID,
      "title":"Hala H273 Private Relative Pixel Comparison T4",
      "code_file":SCRIPT_NAME,"language":"python","kernel_type":"script",
      "is_private":"true","enable_gpu":"true","enable_tpu":"false",
      "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
      "dataset_sources":[],"competition_sources":[],
      "kernel_sources":[],"model_sources":[],
    }
    dest.mkdir(parents=True,exist_ok=False)
    (dest/SCRIPT_NAME).write_text(script,encoding="utf-8")
    (dest/META_NAME).write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"private":True,"slug":KERNEL_ID,"free_gpu":"NvidiaTeslaT4",
            "python_files":1,"model_or_private_media_sources":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--bridge",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(make(a.source,a.bridge,a.destination),sort_keys=True))
