"""H170 private frozen real T4 adapter-forward logit/vision diagnostic."""
import json,os,subprocess,sys
from pathlib import Path
SHA="b61a807965963ade9305d376020087ed7d1a3fe6"
WORK=Path("/kaggle/tmp/h170-code")
OUT=Path("/kaggle/working/hala-h170-private-forward-sensitivity")
JOB=Path("/kaggle/working/h170-private-forward-job.json")

def command(*args):
    subprocess.run(args,capture_output=True,text=True,check=True,timeout=600)

def receipt(status,**data):
    x=dict(schema="hala.h170.private-frozen-visual-forward-kaggle-job.v1",
           status=status,source_commit=SHA,optimizer_steps=0,
           dev_ood_model_calls=0,paid_gpu=False,weights_published=False,
           flagship_35b_unchanged=True,**data)
    JOB.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n","utf8")

def main():
    try:
        root=Path("/kaggle/input")
        files=[p for p in root.rglob("*") if p.is_file()]
        source=[p for p in files if p.name=="h168-training.pkg"]
        frozen=[p for p in files if p.name=="adapter_model.safetensors"
               and "hala-h167-private-frozen-language-only-v1" in p.parts]
        vision=[p for p in files if p.name=="adapter_model.safetensors"
               and "hala-h169-private-vision-only-v1" in p.parts]
        if (len(SHA)!=40 or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or len(files)!=6 or len(source)!=1 or len(frozen)!=1
            or len(vision)!=1
            or "hala-h168-private-train-32-v1" not in source[0].parts
            or {x.name for x in frozen[0].parent.iterdir()}!={"README.md",
                   "adapter_config.json","adapter_model.safetensors"}
            or {x.name for x in vision[0].parent.iterdir()}!={"adapter_config.json",
                   "adapter_model.safetensors"}):
            raise ValueError("H170 only frozen H168 TRAIN, H164 language, H168 vision")
        command(sys.executable,"-m","pip","uninstall","-y","torchao")
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SHA)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(["git","-C",str(WORK),"rev-parse","HEAD"],
                text=True).strip()!=SHA:
            raise ValueError("H170 exact code SHA checkout failed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        receipt("H170_VISUAL_FORWARD_SENSITIVITY_RUNNING",paired_forward_calls=0)
        from h170.sensitivity_gpu import run
        result=run(source[0],frozen[0].parent,vision[0].parent,OUT)
        if result.get("paired_forward_calls")!=2 or result.get("optimizer_steps")!=0:
            raise ValueError("H170 before/after vision forward missing")
        receipt("H170_FROZEN_TRAIN_FORWARD_DIAGNOSTIC_COMPLETE",
                paired_forward_calls=2,peak_allocated_gib=result["peak_allocated_gib"],
                vision_forward_effect_nonzero=result["vision_forward_effect_nonzero"],
                logit_forward_effect_nonzero=result["logit_forward_effect_nonzero"])
    except Exception as e:
        receipt("H170_FORWARD_DIAGNOSTIC_UNQUALIFIED",
                paired_forward_calls=0,
                failure_class=type(e).__name__,failure_message=str(e)[:230])
        raise

if __name__=="__main__":
    main()
