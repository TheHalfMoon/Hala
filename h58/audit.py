"""H58 independently hash-audit genuine original Edge mixed-domain PUBLIC DEV.

Author-created DOM receipts are harness qualification, not neural success.
Never use these PUBLIC DEV scenes or golden coordinates as supervised TRAIN.
"""
from __future__ import annotations
import hashlib,json,struct
from pathlib import Path
SCHEMA="hala.h58.original-multidomain-public-dev.v1"
CAPTURE_SCHEMA="hala.h58.real-edge-multidomain-public-dev.v1"
ORIGINAL_SHA="a1c17c5dca028096bdd653de927fae18b1797a241ca3529646e10a1eb5f4f3de"

def sha(b):return hashlib.sha256(b).hexdigest()
def strict_file(p,max_bytes=1_000_000):
    p=Path(p)
    if p.stat().st_size>max_bytes:raise ValueError("Unbounded original first-party file")
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError("Duplicate JSON key")
            out[k]=v
        return out
    return json.loads(p.read_text("utf8"),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))

def audit(root):
    root=Path(root)
    sb=(root/"source.json").read_bytes()
    if sha(sb)!=ORIGINAL_SHA:raise ValueError("Original firstparty source signature changed")
    source=strict_file(root/"source.json")
    plans=strict_file(root/"plans.json")
    receipt=strict_file(root/"capture.json")
    if (source.get("schema")!=SCHEMA or source.get("split")!="dev" or
        source.get("public_dev_only") is not True or source.get("blind_test") is not False or
        source.get("rights",{}).get("training_allowed") is not False or
        source.get("rights",{}).get("external_images") is not False or
        source.get("rights",{}).get("official_benchmark") is not False or
        source.get("plans_sha256")!=sha((root/"plans.json").read_bytes()) or
        source.get("scenes") is None or len(source["scenes"])!=24 or len(plans)!=24 or
        receipt.get("schema")!=CAPTURE_SCHEMA or
        receipt.get("original_source_sha256")!=ORIGINAL_SHA or
        receipt.get("actual_model_inference") is not False or
        receipt.get("train_rows")!=0 or receipt.get("blind_test") is not False or
        receipt.get("correct_author_oracle_drags")!=24 or
        receipt.get("wrong_start_rejected")!=24 or
        receipt.get("wrong_end_rejected")!=24 or
        receipt.get("unordered_reverse_completed")!=16 or
        receipt.get("ordered_reverse_rejected")!=8):
        raise ValueError("Missing authentic source-captured original PUBLIC DEV identity")
    imgs=set();ids=set();bydomain={};bylanguage={"ar":0,"en":0};order={True:0,False:0}
    scenes={v["id"]:v for v in source["scenes"]}
    captures={v["id"]:v for v in receipt["cases"]}
    if len(scenes)!=24 or len(captures)!=24:raise ValueError("Duplicate original DEV screenshot source")
    for p in plans:
        ident=p["id"]
        if (ident in ids or not ident.startswith("h58-") or ident not in scenes or ident not in captures):
            raise ValueError("Unqualified original scene identity")
        ids.add(ident)
        c=captures[ident]
        if (p.get("kind") not in ("sheet","slide_resize","text_highlight") or
            p.get("language") not in bylanguage or
            p.get("split")!="dev" or p.get("image_size")!=[1024,640] or
            type(p.get("ordered")) is not bool or
            p["ordered"]!=(p["kind"]=="slide_resize") or
            c.get("domain")!=p["kind"] or c.get("language")!=p["language"] or
            c.get("ordered") is not p["ordered"] or
            c.get("image_path")!="png/"+ident+".png" or
            c.get("split")!="dev" or
            c.get("image_size")!=[1024,640]):
            raise ValueError("Original domain and ordered=false semantics changed")
        if any(c.get(k) is not True for k in
                ("bad_start_rejected","bad_end_rejected","original_forward_completed")):
            raise ValueError("Original real GUI negative/positive evidence missing")
        if c.get("original_reverse_completed") is not (not p["ordered"]):
            raise ValueError("Symmetric/ordered real GUI reverse behavior inconsistent")
        if c.get("start_point")!=p["start"] or c.get("end_point")!=p["end"]:
            raise ValueError("Original author-validated GUI endpoints changed")
        html=root/"html"/(ident+".html")
        if sha(html.read_bytes())!=scenes[ident]["html_sha256"] or sha(html.read_bytes())!=c["source_html_sha256"]:
            raise ValueError("Original HTML source edited")
        png=(root/"png"/(ident+".png")).read_bytes()
        if (len(png)<128 or len(png)>5_000_000 or png[:8]!=b"\x89PNG\r\n\x1a\n" or
            struct.unpack(">II",png[16:24])!=(1024,640)):
            raise ValueError("Original screenshot has wrong real Edge PNG dimensions")
        if sha(png)!=c.get("image_sha256") or c["image_sha256"] in imgs:
            raise ValueError("Mutated or duplicated screenshot bytes")
        imgs.add(c["image_sha256"])
        bydomain[p["kind"]]=bydomain.get(p["kind"],0)+1
        bylanguage[p["language"]]+=1
        order[p["ordered"]]+=1
    if (bydomain!={"sheet":8,"slide_resize":8,"text_highlight":8} or
        bylanguage!={"ar":12,"en":12} or order!={True:8,False:16}):
        raise ValueError("Incomplete bilingual source-disjoint domain DEV")
    if len(list((root/"png").glob("*.png")))!=24:
        raise ValueError("Unexpected extra original screenshot")
    return dict(original_source_sha256=ORIGINAL_SHA,actual_original_edge_screens=24,
        ar=12,en=12,domains=bydomain,ordered_true=8,ordered_false=16,
        original_real_browser_correct_drags=24,original_real_browser_wrong_origin_rejected=24,
        original_real_browser_wrong_target_rejected=24,
        original_real_browser_reverse_unordered_accepted=16,
        original_real_browser_reverse_ordered_rejected=8,
        model_neural_inference_performed=False,model_accuracy_measured=False,
        public_dev_only=True,blind_test=False,training_eligible=False)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="h58/assets")
    print(json.dumps(audit(p.parse_args().root),sort_keys=True))
