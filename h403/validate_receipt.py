"""H403: strict private Kaggle SHA receipt, never Hala real-GUI/OOD signoff."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h403-private-no-target-lora"
ADAPTER_PARENT="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537"

def verify(root=ROOT):
    path=Path(root)
    payload=(path/"receipt.json").read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=(path/"receipt.sha256").read_text(encoding="ascii").strip():
        raise ValueError("H403 real Kaggle receipt SHA mismatch")
    def unique(items):
        obj={}
        for k,v in items:
            if k in obj:
                raise ValueError("H403 duplicate JSON key")
            obj[k]=v
        return obj
    r=json.loads(payload,object_pairs_hook=unique)
    if (r.get("schema")!="hala.h403.private-invented-absent-target-lora.v1"
        or r.get("model")!="Qwen/Qwen3-VL-2B-Instruct"
        or r.get("revision")!="89644892e4d85e24eaac8bacfd4f463576704203"
        or r.get("public_base_license")!="apache-2.0"
        or r.get("parent_adapter_sha256")!=ADAPTER_PARENT
        or r.get("train_source")!="originally authored algorithmic rectangles only"
        or r.get("train_uses_prior_h402_eval") is not False
        or r.get("authorized_original_gui_train") is not False
        or r.get("independent_external_ood") is not False
        or r.get("official_hala_model_trained") is not False
        or r.get("weights_published") is not False):
        raise ValueError("H403 illegal real GUI, OOD, or release representation")
    status=r.get("status")
    if status=="COMPLETED_PRIVATE_SYNTHETIC_ABSTENTION_LORA":
        if (r.get("optimizer_steps")!=48
            or r.get("train_n")!=48 or r.get("dev_n")!=24 or r.get("audit_n")!=24
            or r.get("device")!="Tesla T4"
            or type(r.get("changed_trainable_tensors")) is not int
            or r["changed_trainable_tensors"]<=0
            or type(r.get("adapter_sha256")) is not str
            or len(r["adapter_sha256"])!=64
            or type(r.get("adapter_size")) is not int
            or r["adapter_size"]<=1000):
            raise ValueError("H403 incomplete actual private model weight update")
        fingerprints=r.get("split_fingerprints")
        if (type(fingerprints) is not dict or
            set(fingerprints)!={"train","dev","audit_author_visible"} or
            any(type(s) is not str or len(s)!=64 for s in fingerprints.values()) or
            len(set(fingerprints.values()))!=3):
            raise ValueError("H403 invented split integrity incomplete")
        losses=r.get("training_losses")
        if (type(losses) is not list or len(losses)!=48 or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in losses)):
            raise ValueError("H403 training steps not verified")
        for metric in ("parent_dev_loss","adapted_dev_loss"):
            v=r.get(metric)
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:
                raise ValueError("H403 dev loss invalid")
        for group in ("parent_audit_generated","adapted_audit_generated"):
            blocks=r.get(group)
            if type(blocks) is not dict or set(blocks)!={"all","en","ar","positive","absent"}:
                raise ValueError("H403 real native paired measurements missing")
            for key,n in (("all",12),("en",6),("ar",6),("positive",6),("absent",6)):
                v=blocks[key]
                if (type(v) is not dict or set(v)!={"n","correct","false_clicks","invalid","abstain"}
                    or v.get("n")!=n
                    or any(type(v.get(k)) is not int or not 0<=v[k]<=n for k in
                        ("correct","false_clicks","invalid","abstain"))):
                    raise ValueError("H403 generated native metrics invalid")
                if v["correct"]+v["false_clicks"]+v["invalid"]>n:
                    raise ValueError("H403 native diagnosis partition impossible")
            if blocks["positive"]["false_clicks"]!=0:
                raise ValueError("H403 false-click counts only on absent targets")
            for metric in ("correct","false_clicks","invalid","abstain"):
                if (blocks["all"][metric]!=blocks["en"][metric]+blocks["ar"][metric]
                    or blocks["all"][metric]!=blocks["positive"][metric]+blocks["absent"][metric]):
                    raise ValueError("H403 bilingual and absence denominators disagree")
    elif status=="FAILED_PRIVATE_SYNTHETIC_TRAIN_NO_RELEASE":
        if type(r.get("failure_class")) is not str:
            raise ValueError("H403 failure receipt has no error class")
    else:
        raise ValueError("H403 unfinished notebook cannot count as trained")
    return {"sha256":digest,"status":status,
            "scope":"author-invented only, not Hala flagship or independent OOD",
            "parent":r.get("parent_audit_generated"),
            "adapted":r.get("adapted_audit_generated"),
            "adapter_sha256":r.get("adapter_sha256")}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))
