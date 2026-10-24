"""Run private Kaggle H100: actual four-arm NEW strict GUI diagnostic."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="44393dc94caf6aea8861f2c337dd549a74c28635"
REPO=Path("/kaggle/tmp/hala-h100-source")
SCRATCH=Path("/kaggle/tmp/hala-h100-fresh-official-eval")
OUTPUT=Path("/kaggle/working/hala-h100-private-eval")
ATTEMPT=Path("/kaggle/working/hala-h100-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2),"utf8")
    temp.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,
                   timeout=timeout,check=True)

def main():
    data={"schema":"hala.h100.private-new-base-language-vision-attempt.v1",
          "status":"STARTED","stage":"GPU_PREFLIGHT","source_sha":SOURCE_SHA,
          "training_executed":False,"model_published":False,
          "commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("owner-private Kaggle research required")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<1:
            raise ValueError("real available CUDA GPU required")
        data["gpu_names"]=[torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
        data["stage"]="PINNED_QWEN_DEPENDENCIES"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=480)
        pins={"torch":"2.11.0","transformers":"5.16.1",
              "peft":"0.20.0","Pillow":"11.3.0"}
        actual={name:importlib.metadata.version(name) for name in pins}
        if any(actual[n].split("+")[0]!=pins[n] for n in pins):
            raise ValueError("pinned real evaluation runtime changed")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible torchao still installed")
        data["stage"]="PINNED_PUBLIC_SOURCE"
        if REPO.exists():
            raise ValueError("unverified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],
                                   text=True).strip()!=SOURCE_SHA:
            raise ValueError("exact source commit not reproducible")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        sys.path.insert(0,str(REPO))
        data["stage"]="REAL_NEW_FOURWAY_STRICT_OFFICIAL_EVAL"
        record(data)
        from h100.eval import run
        results=run(repo=REPO,work=SCRATCH,output=OUTPUT)
        if (results.get("status")!="REAL_NEW_FOURWAY_BALANCED_VISUAL_VS_OTHERS_STRICT_EVAL"
                or results.get("sample_counts")!={"base":16,"h94_language":16,"h97_vision":16,"h99_balanced_vision":16}
                or not results.get("raw_evidence_sha256")):
            raise ValueError("no paired fresh four-arm benchmark success")
        data.update(status="H100_NEW_FOURWAY_EVAL_COMPLETE",
                    stage="PRIVATE_RAW_SHA_BOUND",
                    strict_successes=results["strict_successes"],
                    raw_evidence_sha256=results["raw_evidence_sha256"])
        record(data)
        print("H100_REAL_NEW_FOURWAY_OFFICIAL_SOURCE_EVAL_COMPLETE",flush=True)
    except Exception as error:
        data.update(status="FAILED_NO_QUALIFIED_FOURWAY_GUI_EVAL",
                    failure_class=type(error).__name__)
        record(data)
        print("HALA_H100_EVAL_FAILED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()
