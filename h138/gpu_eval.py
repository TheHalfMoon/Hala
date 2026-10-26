"""H138 frozen two-complete-app eval, zero optimizer steps."""
import json
from pathlib import Path
from h138.source import SCHEMA,digest,validate_source
from h136.train_gpu import BASE_MODEL,BASE_REV,model_predict
from h136.prompt import role_prompt

SOURCE_SHA="203e5976131e685e296c4cf71edffe3c4332d9103fb7d508b041a850ce07e7c7"
ADAPTER_SHA="183680c3224b024c546dbbd1428e7a05ffa1491d57c117aec350a11e04cba1a7"
SCHEMA_OUT="hala.h138.owner-two-untouched-full-MIT-kanban-apps-real-neural.v1"

def inputs():
    root=Path("/kaggle/input")
    source=[];adapters=[]
    for file in root.rglob("manifest.json"):
        try:record=json.loads(file.read_text("utf8"))
        except (ValueError,OSError,TypeError,json.JSONDecodeError):continue
        if record.get("schema")!=SCHEMA:continue
        if digest(file.read_bytes())!=SOURCE_SHA:
            raise ValueError("H138 original source SHA changed")
        source.append((file.parent,validate_source(file.parent)))
    for f in root.rglob("adapter_model.safetensors"):
        if digest(f.read_bytes())==ADAPTER_SHA and (
            f.parent/"adapter_config.json").is_file():
            adapters.append(f.parent)
    if len(source)!=1 or len(adapters)!=1:
        raise ValueError("H138 exactly one full-app source suite and frozen adapter")
    return source[0],adapters[0]

def evaluate(output):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H138 no paid GPU, private FREE T4 required")
    (root,manifest),adapter_dir=inputs()
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                              trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    results=[]
    metrics={}
    for arm in ("base_pretrained","h136_adapter"):
        if arm=="h136_adapter":
            model=PeftModel.from_pretrained(model,adapter_dir,
                                             is_trainable=False).to("cuda:0").eval()
        for case in manifest["cases"]:
            with Image.open(root/case["image"]) as im:
                if im.format!="PNG" or im.size!=(960,720):
                    raise ValueError("H138 genuine full-app screenshot required")
                frame=im.convert("RGB")
            for role in ("start","end"):
                prompt=role_prompt(case["instruction"],role)
                raw,p=model_predict(processor,model,torch,frame,prompt)
                b=case[role+"_bbox"]
                hit=(p is not None and b["x"]<=p[0]<=b["x"]+b["width"]
                     and b["y"]<=p[1]<=b["y"]+b["height"])
                segment=arm+"__"+case["app"]+"__"+case["lang"]+"__"+role
                scores=metrics.setdefault(segment,{"hits":0,"valid":0})
                scores["hits"]+=int(hit)
                scores["valid"]+=int(p is not None)
                results.append({"arm":arm,"case_id":case["case_id"],
                                "image_sha256":case["image_sha256"],
                                "role":role,"raw":raw,"point":p,"hit":bool(hit)})
    if len(results)!=32:raise ValueError("H138 frozen 32 calls incomplete")
    report={"schema":SCHEMA_OUT,
      "status":"H138_PRIVATE_2_FULL_UPSTREAM_APPS_32_NEURAL_COMPLETE",
      "source_sha256":SOURCE_SHA,"adapter_sha256":ADAPTER_SHA,
      "source_app_revisions":manifest["upstream_git_shas"],
      "actual_neural_calls":32,"actual_optimizer_steps":0,
      "app_scenes":8,"paid_compute":False,
      "main_Hala_flagship_weights_changed":False,
      "scores":metrics,
      "real_full_app_browser_success_not_yet_scored":True}
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-h138-neural.json").write_text(
      json.dumps(results,sort_keys=True,indent=2)+"\n","utf8")
    (output/"private-h138-report.json").write_text(
      json.dumps(report,sort_keys=True,indent=2)+"\n","utf8")
    return report
