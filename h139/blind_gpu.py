"""One-shot, PRIVATE, inference-only H139 unseen React application benchmark.

The uploaded screenshot manifest is redacted: NO START/END source geometry,
DOM state, reference mouse actions or other oracle labels are present.
"""
import json
from pathlib import Path
from zipfile import ZipFile
from h139.blind_export import SCHEMA as INPUT_SCHEMA,FIELDS
from h139.owner_replay import NEURAL_SCHEMA,ARMS
from h139.registry import sha,UPSTREAM
from h139.prompt import role_prompt
from h139.train_gpu import BASE_MODEL,BASE_REV,model_predict

ADAPTER_SHA="c46ec0abf4d9fed545ec85867737ca19dc59f54f84e847b90304d4af6b052180"
ARCHIVE_SHA="7d3ddcb6db22c4fc1974c099a318e8921adcdfc514aae8de0bd37444ed055fc8"
REDACTED_MANIFEST_SHA="3432b51aa505688b513210ea3ae4dc34ff3d514790f671b532fc84cbc3c83c73"
OWNER_MANIFEST_SHA="c26c0b1c850bf4fc30c7e04bc16e547f89d69269174751f26ea2b6b824acfac9"

def admit(root=None):
    root=Path("/kaggle/input" if root is None else root)
    zips=list(root.rglob("blind8-redacted-eval-only.zip"))
    adapter=list(root.rglob("adapter_model.safetensors"))
    if len(zips)!=1 or len(adapter)!=1:
        raise ValueError("H139 requires exactly one private redacted BLIND and one frozen adapter")
    if (sha(zips[0].read_bytes())!=ARCHIVE_SHA or
        sha(adapter[0].read_bytes())!=ADAPTER_SHA or
        not (adapter[0].parent/"adapter_config.json").is_file()):
        raise ValueError("H139 frozen private inference payload SHA drift")
    forbidden=("train64","dev16","sealed","train896","eval128")
    mounts=[x.name.lower() for x in root.iterdir() if x.is_dir()]
    # Kaggle may expose additional system mount directories. Admit exactly
    # one byte-pinned blind archive and one byte-pinned adapter above; do
    # not rely on an exact directory count for the hosted input root.
    if any(any(f in m for f in forbidden) for m in mounts):
        raise ValueError("H139 blind neural inference only; reject any train/dev/old dataset")
    with ZipFile(zips[0]) as z:
        entries=z.namelist()
        if len(entries)!=9 or len(entries)!=len(set(entries)) or any(
                "/" in f or "\\" in f or ".." in f for f in entries):
            raise ValueError("H139 redacted input ZIP duplicate/traversal")
        raw=z.read("manifest.json")
        if sha(raw)!=REDACTED_MANIFEST_SHA:
            raise ValueError("H139 redacted frozen manifest changed")
        doc=json.loads(raw)
        if (doc.get("schema")!=INPUT_SCHEMA or doc.get("private") is not True or
            doc.get("source_role")!="blind_test" or
            doc.get("training_eligible") is not False or
            doc.get("original_manifest_sha256")!=OWNER_MANIFEST_SHA or
            doc.get("source_app_sha")!=UPSTREAM["ood-board"]["git_sha"] or
            doc.get("case_count")!=8 or len(doc.get("cases",[]))!=8):
            raise ValueError("H139 source not frozen unseen external-app OOD")
        case_hashes=set()
        for row in doc["cases"]:
            if (set(row)!=set(FIELDS) or row["app"]!="ood-board" or
                row["app_git_sha"]!=UPSTREAM["ood-board"]["git_sha"] or
                row["lang"] not in ("en","ar") or row["image"] not in entries or
                not row["instruction"] or
                len(row["image_sha256"])!=64):
                raise ValueError("H139 unauthorized source/label field or missing native input")
            pixels=z.read(row["image"])
            if sha(pixels)!=row["image_sha256"] or row["image_sha256"] in case_hashes:
                raise ValueError("H139 duplicate or tampered native blind screenshot")
            case_hashes.add(row["image_sha256"])
        if set(entries)!={"manifest.json"}|{x["image"] for x in doc["cases"]}:
            raise ValueError("H139 inference ZIP contains unexpected private files")
    return zips[0],doc,adapter[0].parent

def run(output):
    import torch,transformers
    from PIL import Image
    from io import BytesIO
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H139 one-shot private FREE T4 inference only")
    archive,doc,adapter_dir=admit()
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                              trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    results=[]
    with ZipFile(archive) as z:
        for arm in ARMS:
            if arm=="h139_adapter":
                model=PeftModel.from_pretrained(model,adapter_dir,
                                                 is_trainable=False).to("cuda:0").eval()
            for row in doc["cases"]:
                with Image.open(BytesIO(z.read(row["image"]))) as image:
                    if image.format!="PNG" or image.size!=(960,720):
                        raise ValueError("H139 original native 960x720 OOD screenshot only")
                    frame=image.convert("RGB")
                for role in ("start","end"):
                    prompt=role_prompt(row["instruction"],role)
                    raw,point=model_predict(processor,model,torch,frame,prompt)
                    results.append({"arm":arm,"id":row["id"],
                                    "image_sha256":row["image_sha256"],
                                    "role":role,"raw":raw,"point":point})
    if len(results)!=32:
        raise ValueError("H139 complete 32 model calls required")
    receipt={"schema":NEURAL_SCHEMA,
             "status":"H139_COMPLETE_32_REAL_BLIND_NEURAL_CALLS",
             "original_owner_manifest_sha256":OWNER_MANIFEST_SHA,
             "redacted_manifest_sha256":REDACTED_MANIFEST_SHA,
             "adapter_sha256":ADAPTER_SHA,"model_calls":32,
             "optimizer_steps":0,"source_role":"blind_test",
             "paid_compute_used":False,"main_35B_updated":False,
             "blind_gold_boxes_available_to_model":False,
             "actual_browser_success_not_yet_scored":True}
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-h139-blind-report.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf8")
    (output/"private-h139-blind-neural.json").write_text(
        json.dumps(results,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return receipt
