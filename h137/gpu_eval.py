"""H137 first independent MIT library-backed GUI benchmark (NO training).

36 heldout EN+AR tasks from real noUiSlider, Split.js, SortableJS runtimes.
Fixed external base Qwen3.5-2B vs frozen H136 rank8 research adapter.
Only private FREE T4. Screenshot/bbox gold never enters model prompt.
"""
import json
from pathlib import Path
from h136.train_gpu import BASE_MODEL,BASE_REV,model_predict
from h136.prompt import role_prompt
from h137.source import SCHEMA,check,digest

FROZEN_SOURCE_MANIFEST_SHA="c9762ca1a1ed4251cf42caba07199e4bbf5ed137a063b68619f79f00085fa324"
FROZEN_ADAPTER_SHA="183680c3224b024c546dbbd1428e7a05ffa1491d57c117aec350a11e04cba1a7"
OUTPUT_SCHEMA="hala.h137.private-real-third-party-library-first-blind-eval.v1"

def admit():
    root=Path("/kaggle/input")
    if not root.is_dir():raise ValueError("H137 private sources unavailable")
    source=[]
    adapters=[]
    for f in root.rglob("manifest.json"):
        try:m=json.loads(f.read_text("utf8"))
        except (OSError,ValueError,TypeError,json.JSONDecodeError):continue
        if m.get("schema")!=SCHEMA:continue
        if (m.get("evaluation_only") is not True or
            m.get("training_or_selection_eligible") is not False):
            raise ValueError("H137 original library source must remain eval-only")
        folder=f.parent
        if digest(f.read_bytes())!=FROZEN_SOURCE_MANIFEST_SHA:
            raise ValueError("H137 frozen first foreign library source changed")
        source.append((folder,check(folder)))
    for f in root.rglob("adapter_model.safetensors"):
        if digest(f.read_bytes())==FROZEN_ADAPTER_SHA and (
            f.parent/"adapter_config.json").is_file():
            adapters.append(f.parent)
    if len(source)!=1 or len(adapters)!=1:
        raise ValueError("H137 frozen source and adapter exact one required")
    return source[0],adapters[0]

def run(output):
    import torch,transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText
    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H137 exactly fresh owner-private FREE Kaggle T4")
    (source_dir,source),adapter_dir=admit()
    if len(source["cases"])!=36:
        raise ValueError("H137 36 independent vendor engine scenes")
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                            trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    report={}
    raw=[]
    for arm in ("pretrained","h136_64step_adapter"):
        if arm=="h136_64step_adapter":
            model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False).to("cuda:0").eval()
        tally={}
        for case in source["cases"]:
            with Image.open(source_dir/case["image"]) as im:
                if im.format!="PNG" or im.size!=(960,720):
                    raise ValueError("H137 genuine 960x720 browser screenshot required")
                rgb=im.convert("RGB")
            category=case["family"]+"__"+case["lang"]
            if category not in tally:
                tally[category]={"start":{"hits":0,"valid":0},"end":{"hits":0,"valid":0}}
            for role in ("start","end"):
                prompt=role_prompt(case["instruction"],role)
                response,point=model_predict(processor,model,torch,rgb,prompt)
                b=case[role+"_bbox"]
                hit=(point is not None and
                     b["x"]<=point[0]<=b["x"]+b["width"] and
                     b["y"]<=point[1]<=b["y"]+b["height"])
                tally[category][role]["hits"]+=int(hit)
                tally[category][role]["valid"]+=int(point is not None)
                raw.append({
                    "arm":arm,"case_id":case["case_id"],"lang":case["lang"],
                    "family":case["family"],"image_sha256":case["image_sha256"],
                    "role":role,"raw":response,"point":point,"hit":bool(hit)})
        report[arm]=tally
    if len(raw)!=144:raise ValueError("H137 first 36 x2 arms x2 roles must run fully")
    result={
        "schema":OUTPUT_SCHEMA,
        "status":"H137_PRIVATE_THREE_REAL_MIT_INTERACTION_LIBRARIES_144_INFERENCE",
        "actual_neural_calls":144,"actual_optimizer_steps":0,
        "new_eval_images":36,"english_cases":18,"arabic_cases":18,
        "vendor_engine_families":3,"source_sha256":FROZEN_SOURCE_MANIFEST_SHA,
        "frozen_adapter_sha256":FROZEN_ADAPTER_SHA,
        "base_model":BASE_MODEL,"base_revision":BASE_REV,
        "paid_compute_used":False,"training_source_attached":False,
        "benchmark_is_independent_production_apps":False,
        "main_Hala_flagship_35B_weights_updated":False,
        "real_mouse_task_success_pending_owner_replay":True,
        "scores":report}
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-h137-raw-neural.json").write_text(
        json.dumps(raw,sort_keys=True,indent=2)+"\n","utf8")
    (output/"private-h137-neural-results.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
