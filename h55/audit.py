"""Independent source-bound audit of actual Edge H55 public DEV screenshots.

This proves only real browser mock-slider geometry and clean source provenance,
NOT trained Hala action prediction, blind task success, or benchmark standing.
"""
from pathlib import Path
import hashlib,json,math,struct

SCHEMA="hala.h55.original-bilingual-drag-public-dev.v1"
CAPTURE="hala.h55.real-edge-native-drag-public-dev-replay.v1"
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return (json.dumps(x,ensure_ascii=False,indent=2)+"\n").encode("utf8")
def audit(directory):
    root=Path(directory)
    source=json.loads((root/"source.json").read_text("utf8"))
    plans=json.loads((root/"plans.json").read_text("utf8"))
    receipt=json.loads((root/"capture.json").read_text("utf8"))
    if (source.get("schema")!=SCHEMA or source.get("split")!="dev" or
        source.get("training_eligible") is True or source.get("public_dev_only") is not True or
        source.get("blind_test") is not False or source.get("trained_model_predictions") not in (None,0) or
        source.get("rights",{}).get("train_eligible") is not False or
        source.get("plans_sha256")!=sha(canon(plans)) or
        receipt.get("schema")!=CAPTURE or receipt.get("source_sha256")!=sha(canon(source)) or
        len(plans)!=24 or len(source["scenes"])!=24 or len(receipt["cases"])!=24):
        raise ValueError("H55 original PUBLIC DEV source/receipt mismatch")
    if (receipt.get("real_browser_correct_drags")!=24 or
        receipt.get("real_browser_wrong_endpoint_rejections")!=24 or
        receipt.get("real_browser_wrong_origin_rejections")!=24 or
        receipt.get("source_train")!=0 or receipt.get("real_model_inference") is not False or
        receipt.get("blind_benchmark_test") is not False):
        raise ValueError("H55 actual browser replay metrics did not qualify")
    language={"ar":0,"en":0}; pngs=set(); families=set()
    for plan,scene,case in zip(plans,source["scenes"],receipt["cases"]):
        id=plan["id"];families.add(plan["family"])
        if (case["id"]!=id or scene["id"]!=id or case["family"]!=plan["family"] or
            case["language"]!=plan["language"] or case["split"]!="dev" or
            case["image_path"]!="png/"+id+".png" or
            plan["viewport"]!=[1024,640] or
            case["source_image_dimensions"]!=[1024,640] or
            case["goal_percent"]!=plan["goal_percent"] or
            case["source_html_sha256"]!=scene["html_sha256"]):
            raise ValueError("H55 source versus browser capture drift "+id)
        lang=case["language"]
        if lang not in language:raise ValueError("Unknown bilingual language")
        language[lang]+=1
        doc=(root/"html"/(id+".html")).read_bytes()
        if sha(doc)!=scene["html_sha256"] or b"pointerdown" not in doc or b"pointermove" not in doc:
            raise ValueError("H55 authored genuine drag handler altered")
        png=(root/"png"/(id+".png")).read_bytes()
        if (not png.startswith(b"\x89PNG\r\n\x1a\n") or
            struct.unpack(">II",png[16:24])!=(1024,640) or
            case["image_sha256"]!=sha(png)):
            raise ValueError("Original Edge screenshot changed "+id)
        if case["image_sha256"] in pngs:raise ValueError("Duplicate screenshot")
        pngs.add(case["image_sha256"])
        t=case["track_bbox"];h=case["handle_bbox"];start=case["start_point"];end=case["end_point"]
        if (len(t)!=4 or len(h)!=4 or len(start)!=2 or len(end)!=2 or
            not all(type(x) in (int,float) and math.isfinite(x) for x in t+h+start+end) or
            not (h[0]<=start[0]<=h[0]+h[2] and h[1]<=start[1]<=h[1]+h[3]) or
            abs(end[0]-(t[0]+t[2]*plan["goal_percent"]/100))>1.5 or
            end[1]!=start[1] or not 0<=end[0]<1024 or not 0<=end[1]<640 or
            end[0]-start[0]<40 or
            case["wrong_origin_rejected"] is not True or
            case["wrong_endpoint_rejected"] is not True or
            case["correct_drag_completed"] is not True or
            case["model_inference"] is not False):
            raise ValueError("H55 original real browser geometric/postcondition integrity "+id)
    if language!={"ar":12,"en":12} or len(families)!=6 or len(pngs)!=24:
        raise ValueError("Missing app-disjoint bilingual DEV diversity")
    return dict(schema=SCHEMA,actual_first_party_public_dev=24,language=language,
       distinct_original_png=len(pngs),families=len(families),
       genuine_browser_accepted_drags=24,real_wrong_end_rejected=24,
       real_wrong_origin_rejected=24,training_screens=0,
       public_dev_not_blind_test=True,model_inference_executed=False,
       original_source_sha256=receipt["source_sha256"],
       browser_receipt_sha256=sha((root/"capture.json").read_bytes()))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="h55/assets")
    print(json.dumps(audit(p.parse_args().root),sort_keys=True))
