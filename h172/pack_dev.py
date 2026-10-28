"""H172 private frozen DEV model-input ZIP: original pixels + task only, NO gold."""
import argparse,hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path
from PIL import Image
from h172.source_dev import SCHEMA,cases

ZIP_SCHEMA="hala.h172.private-new-original-dev-only-gpu-input.v1"

def sha(b):return hashlib.sha256(b).hexdigest()

def pack(source,target):
    source=Path(source);target=Path(target)
    if target.exists() or target.suffix!=".zip":
        raise ValueError("H172 fresh DEV model ZIP must be new and private")
    doc=json.loads((source/"manifest.json").read_text("utf8"))
    rows=doc.get("private_cases")
    expected=cases()
    if (doc.get("schema")!=SCHEMA or doc.get("split")!="DEV"
        or doc.get("count")!=32
        or doc.get("owner_gold_uploaded") is not False
        or doc.get("previous_consumed_sources_excluded") is not True
        or not isinstance(rows,list) or len(rows)!=32):
        raise ValueError("H172 real original DEV source integrity invalid")
    seen=set();images={};outputs=[]
    for prior,row in zip(expected,rows):
        if (not all(row.get(k)==v for k,v in prior.items())
            or row.get("owner_native_reference_passed") is not True
            or row.get("image")!="images/"+str(row["index"])+".png"
            or row.get("image_sha256") in seen
            or any(k in row for k in (
                "owner_gold","training_supervision","reference_point",
                "owner_native_actions","owner_reference_action"))
            or not row.get("instruction")):
            raise ValueError("H172 source DEV contains gold/old/missing original case")
        png=(source/row["image"]).read_bytes()
        if sha(png)!=row["image_sha256"]:
            raise ValueError("H172 original DEV frozen PNG bytes mutated")
        with Image.open(io.BytesIO(png)) as im:
            if im.format!="PNG" or im.size!=(960,720):
                raise ValueError("H172 original DEV 960x720 PNG expected")
        images[row["image"]]=png
        seen.add(row["image_sha256"])
        outputs.append({k:row[k] for k in (
            "index","split","app","variant","action_type","task","language",
            "image","image_sha256","instruction")})
    if (Counter(r["language"] for r in outputs)!={"ar":16,"en":16}
        or Counter(r["app"] for r in outputs)!={"flatpickr":16,"awesomplete":16}
        or len(images)!=32):
        raise ValueError("H172 DEV app/language source quota not met")
    record=dict(schema=ZIP_SCHEMA,split="DEV",count=32,
        reference_gold_uploaded=False,
        previous_train_and_consumed_eval_excluded=True,private_cases=outputs)
    target.parent.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(
            record,sort_keys=True,separators=(",",":"),ensure_ascii=False))
        for path,png in sorted(images.items()):
            z.writestr(path,png)
    return dict(status="H172_PRIVATE_NEW_DEV_ONLY_INPUTS_FROZEN",
       cases=32,arabic=16,english=16,owner_gold_uploaded=False,
       optimizer_steps=0,dev_zip_sha256=sha(target.read_bytes()))

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--source",required=True)
    a.add_argument("--output",required=True)
    args=a.parse_args()
    print(json.dumps(pack(args.source,args.output),indent=2,sort_keys=True))
