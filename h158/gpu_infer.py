"""H158 independent different-app OOD model inference, free private T4 only.

32 genuine GPU calls (16 new original TodoMVC screenshots × 2 frozen
research LoRAs), zero optimizer/training. Full browser success is checked
later, OWNER ONLY, not inferred from valid JSON.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from h156.train_gpu import open_media,inference
from h156.action_contract_v2 import action_from_dict_v2
from h158.audit_source import MANIFEST_SHA
from h158.original_todomvc_source import COUNTS,TOTAL,PIN,SCHEMA
from h158.pack_private_ood import SHA_H156,SHA_H157,FILES

SRC_ZIP_SHA="d870943c02c6449d61184a19d4d49c708b85577e0d090065d81e95304d899a3c"
ROLES=("before","after")
MODEL_SHAS={"before":SHA_H156,"after":SHA_H157}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_cases(root):
    root=Path(root)
    if not (root.is_dir() or root.is_file()):
        raise ValueError("H158 owner private source missing")
    if root.is_file() and sha(root.read_bytes())!=SRC_ZIP_SHA:
        raise ValueError("H158 original source ZIP SHA not verified")
    with open_media(root) as pack:
        names=pack.namelist()
        if "manifest.json" not in names or sha(pack.read("manifest.json"))!=MANIFEST_SHA:
            raise ValueError("H158 original frozen source manifest mismatch")
        doc=json.loads(pack.read("manifest.json"))
        cases=doc.get("private_cases")
        if (doc.get("schema")!=SCHEMA or doc.get("original_upstream_sha")!=PIN or
            doc.get("model_inference_calls")!=0 or
            doc.get("real_app_state_correct")!=TOTAL or
            len(cases)!=TOTAL or
            set(names)!={"manifest.json"}|{"images/"+c["image"] for c in cases}):
            raise ValueError("H158 external app cases/paths changed")
        seen=set()
        counts=Counter()
        for case in cases:
            data=pack.read("images/"+case["image"])
            digest=sha(data)
            if digest!=case["image_sha256"] or digest in seen:
                raise ValueError("H158 original snapshot duplicated/altered")
            seen.add(digest)
            counts[(case["task"],case["language"])]+=1
        expected={(t,lang):n for t,groups in COUNTS.items()
                  for lang,n in groups.items()}
        if counts!=expected or len(seen)!=TOTAL:
            raise ValueError("H158 required bilingual OOD split incomplete")
    return cases


def import_adapter(archive,dest,expected_sha):
    archive=Path(archive)
    dest=Path(dest)
    if dest.exists():
        raise FileExistsError("H158 model extraction is create-only")
    with open_media(archive) as z:
        if set(z.namelist())!=FILES:
            raise ValueError("H158 source-adapter archive has unexpected files")
        if sha(z.read("adapter_model.safetensors"))!=expected_sha:
            raise ValueError("H158 model weight SHA differs from frozen H156/H157")
        config=json.loads(z.read("adapter_config.json"))
        if config.get("r")!=8:
            raise ValueError("H158 requires frozen rank8 research LoRA")
        dest.mkdir(parents=True,exist_ok=False)
        for name in sorted(FILES):
            (dest/name).write_bytes(z.read(name))
    return dest


def validated_model_action(raw):
    if not isinstance(raw,str) or len(raw)>1600:
        return {"valid":False,"click":None,"abstain":False}
    try:
        data=json.loads(raw)
        if not isinstance(data,dict) or set(data)!={"actions"}:
            raise ValueError("Expected model actions object")
        actions=data["actions"]
        if not isinstance(actions,list) or len(actions)!=1:
            raise ValueError("H158 click-only one action")
        action=action_from_dict_v2(actions[0])
        if action.kind=="abstain":
            return {"valid":True,"click":None,"abstain":True}
        if action.kind!="click" or action.point is None:
            raise ValueError("External TodoMVC OOD tasks allow only click(point)")
        return {"valid":True,"click":list(action.point),"abstain":False}
    except (ValueError,TypeError,KeyError):
        return {"valid":False,"click":None,"abstain":False}


def run(source,old_archive,new_archive,output):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    from h132.grounder_contract import MODEL_ID,MODEL_SHA

    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H158 independent OOD inference only on free private T4")
    cases=source_cases(source)
    old=import_adapter(old_archive,Path("/kaggle/working/h158-prior-lora"),SHA_H156)
    trained=import_adapter(new_archive,Path("/kaggle/working/h158-new-lora"),SHA_H157)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(
        base,str(old),adapter_name="prior",is_trainable=False)
    model.load_adapter(str(trained),adapter_name="current",is_trainable=False)
    model.eval()
    responses=[]
    counts=Counter()
    with open_media(source) as archive:
        for role,adapter_name in (("before","prior"),("after","current")):
            model.set_adapter(adapter_name)
            for case in cases:
                with Image.open(io.BytesIO(archive.read(
                      "images/"+case["image"]))) as img:
                    picture=img.convert("RGB")
                if picture.size!=(960,720):
                    raise ValueError("H158 model OOD image not native resolution")
                raw=inference(processor,model,torch,picture,case)
                proposed=validated_model_action(raw)
                counts[role+"_calls"]+=1
                counts[role+"_valid"]+=int(proposed["valid"])
                counts[role+"_click"]+=int(proposed["click"] is not None)
                responses.append({
                    "role":role,"task":case["task"],"language":case["language"],
                    "image_sha":case["image_sha256"],
                    "raw":raw,"model_action":proposed})
                if len(responses)%8==0:
                    print("H158_REAL_FREE_T4_NEW_APP_CALLS",
                          len(responses),32,flush=True)
    if counts["before_calls"]!=16 or counts["after_calls"]!=16:
        raise ValueError("H158 full paired external OOD GPU calls incomplete")
    result={
        "schema":"hala.h158.private-external-todomvc-gpu-inference.v1",
        "status":"H158_COMPLETE_REAL_32_GPU_NEW_APP_INFERENCES",
        "different_untrained_original_app":"tastejs/todomvc",
        "external_app_sha":PIN,"source_manifest_sha256":MANIFEST_SHA,
        "older_research_lora_sha":SHA_H156,
        "newer_research_lora_sha":SHA_H157,
        "total_gpu_calls":len(responses),
        "counts":dict(counts),
        "optimizer_steps":0,
        "native_app_state_success_measured":False,
        "h140_blind_calls":0,"paid_gpu":False,
        "Hala35B_unchanged":True,
        "model_weights_published":False,
        "private_predictions":responses,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"h158-private-external-ood-predictions.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in result.items() if k!="private_predictions"}
