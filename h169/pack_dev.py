"""Freeze owner-only NEW H169 DEV model inputs, NO owner gold on GPU."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from PIL import Image
from h169.source_dev import SCHEMA,COUNT,cases,admitted_history

def sha(b):return hashlib.sha256(b).hexdigest()

def pack(owner,source,train_source,destination):
    source=Path(source);dest=Path(destination)
    if dest.exists() or dest.suffix!=".zip":
        raise ValueError("H169 DEV model input must be a new private immutable ZIP")
    doc=json.loads((source/"manifest.json").read_text("utf8"))
    expected=cases()
    prior=admitted_history(owner,train_source)
    records=doc.get("private_cases")
    if (doc.get("schema")!=SCHEMA or doc.get("split")!="DEV"
        or doc.get("count")!=COUNT or doc.get("reference_gold_uploaded") is not False
        or not isinstance(records,list) or len(records)!=COUNT):
        raise ValueError("H169 DEV source identity not frozen")
    seen=set(prior); images={}
    for base,row in zip(expected,records):
        if (not all(row.get(k)==v for k,v in base.items())
            or row.get("image_sha256") in seen
            or row.get("image")!="images/"+str(base["index"])+".png"
            or row.get("native_reference_checked") is not True
            or any(k in row for k in (
                "training_supervision","owner_gold","owner_reference_action",
                "reference_point","gold","owner_native_actions"))):
            raise ValueError("H169 DEV cannot contain owner supervision/history")
        pic=(source/row["image"]).read_bytes()
        if sha(pic)!=row["image_sha256"]:
            raise ValueError("H169 image SHA mismatch")
        with Image.open(io.BytesIO(pic)) as f:
            if f.size!=(960,720) or f.format!="PNG":
                raise ValueError("H169 original DEV screenshot shape/form invalid")
        images[row["image"]]=pic
        seen.add(row["image_sha256"])
    subset={"schema":"hala.h169.private-new-original-dev-only-gpu-input.v1",
        "split":"DEV","count":COUNT,"reference_gold_uploaded":False,
        "historical_train_ood_excluded":True,
        "source_app_families":["choices","nouislider"],
        "private_cases":[{k:v for k,v in r.items() if k in (
          "index","split","app","variant","action_type","task","language",
          "image","image_sha256","instruction")} for r in records]}
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,"x",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(
            subset,sort_keys=True,ensure_ascii=False,separators=(",",":")))
        for key,val in sorted(images.items()):
            z.writestr(key,val)
    return dict(status="H169_PRIVATE_32_DEV_INPUTS_LOCKED",
      dev_zip_sha256=sha(dest.read_bytes()),images=COUNT,
      english_cases=16,arabic_cases=16,owner_gold_uploaded=False,
      optimizer_steps=0,previous_dev_ood_used=False)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("owner","source","train-source","dest"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    print(json.dumps(pack(a.owner,a.source,a.train_source,a.dest),indent=2,sort_keys=True))
