"""H164 owner-private Kaggle pack, creates only from all fully qualified sources.

TRAIN 384 images with labels; DEV 96 and OOD 64 images/tasks with NO owner
reference or gold points. No model weights published or paid provider allowed.
"""
import argparse,json,shutil,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from h164.audit_full_qualify import audit
from h164.expand_288_train import digest

PILOT="private-h164/four-app-train-pilot-v1"
EXTRA="private-h164/additional-three-app-288-train-v1"
DEV="private-h164/new-dev-96-v1"
OOD="private-h164/new-two-app-ood-64-v1"
DEST="private-h164/kaggle-h164-real-original-ui-v1"
MODEL_SHA="72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c"
MODEL_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"
DATASET="abdulazizshehri/hala-h164-private-real-original-ui-v1"

def build(owner):
    owner=Path(owner)
    qualification=audit(owner)
    if qualification.get("status")!=(
        "H164_FINAL_384_TRAIN_96_DEV_64_TRUE_TWO_NEW_APP_OOD_ADMITTED"
    ):
        raise ValueError("H164 real new original OOD not independently qualified")
    recipes=[
        ("TRAIN",[(PILOT,96),(EXTRA,288)],384),
        ("DEV",[(DEV,96)],96),
        ("OOD",[(OOD,64)],64)
    ]
    base=owner/DEST
    if base.exists():
        raise FileExistsError("H164 all data pack create-only")
    orig_model=owner/"private-h162/kaggle-eval-v1/h161-mixed-trained-research-lora.zip"
    if digest(orig_model.read_bytes())!=MODEL_ZIP_SHA:
        raise ValueError("H164 parent external 2B rank8 LoRA archive changed")
    with ZipFile(orig_model) as z:
        if digest(z.read("adapter_model.safetensors"))!=MODEL_SHA:
            raise ValueError("H164 parent adapter weights changed")
    base.mkdir(parents=True,exist_ok=False)
    archives={}
    all_images=set()
    for split,roots,total in recipes:
        cases=[]
        pairs=[]
        for relative,n in roots:
            root=owner/relative
            doc=json.loads((root/"manifest.json").read_text("utf8"))
            records=doc["private_cases"]
            if len(records)!=n:
                raise ValueError("H164 source manifest count/source drift")
            for row in records:
                safe=dict(row)
                if split!="TRAIN":
                    banned=("training_supervision","owner_gold_click",
                            "owner_supervision","reference_point",
                            "owner_reference_action")
                    if any(b in safe for b in banned):
                        raise ValueError("H164 original DEV/OOD reference gold leaked")
                else:
                    if not isinstance(safe.get("training_supervision"),dict):
                        raise ValueError("H164 original app TRAIN gold missing")
                image=safe["image"]
                target=split.lower()+"-"+image
                safe["image"]=target
                blob=(root/"images"/image).read_bytes()
                if digest(blob)!=row["image_sha256"]:
                    raise ValueError("H164 original browser PNG SHA changed")
                if row["image_sha256"] in all_images:
                    raise ValueError("H164 same image across split or source")
                all_images.add(row["image_sha256"])
                pairs.append((target,blob))
                cases.append(safe)
        if len(cases)!=total:
            raise ValueError("H164 frozen training/eval example count changed")
        export={"schema":"hala.h164.private-"+split.lower()+"-source.v1",
                "split":split,"count":total,"reference_gold_uploaded":split=="TRAIN",
                "source_qualification":"H164_FINAL_384_TRAIN_96_DEV_64_TRUE_TWO_NEW_APP_OOD_ADMITTED",
                "private_cases":cases,
                "h140_blind_calls":0,"paid_GPU":False}
        filename=base/("h164-"+split.lower()+"-real-original-ui.zip")
        with ZipFile(filename,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
            z.writestr("manifest.json",json.dumps(
                export,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
            for name,blob in pairs:
                z.writestr("images/"+name,blob)
        with ZipFile(filename) as z:
            raw=z.read("manifest.json")
            if z.testzip() is not None or len(z.namelist())!=total+1:
                raise ValueError("H164 private source ZIP incomplete/corrupted")
            if split!="TRAIN" and (
                b"training_supervision" in raw or
                b"owner_supervision" in raw or
                b"owner_gold_click" in raw or b"reference_point" in raw):
                raise ValueError("H164 evaluation source leaked gold")
        archives[split]={"name":filename.name,"sha256":digest(filename.read_bytes()),
                         "count":total}
    shutil.copyfile(orig_model,base/"h164-parent-h161-frozen-lora.zip")
    (base/"dataset-metadata.json").write_text(
        json.dumps({"id":DATASET,
                    "title":"hala-h164-private-real-original-ui-v1",
                    "licenses":[{"name":"other"}]},
                   indent=2)+"\n","utf8")
    receipt={"status":"H164_PRIVATE_384_96_64_SOURCE_ZIPS_READY",
             "qualification":qualification["status"],
             "archives":archives,"source_images":len(all_images),
             "baseline_model_sha256":MODEL_SHA,
             "DEV_gold_uploaded":False,"OOD_gold_uploaded":False,
             "optimizer_steps":0,"paid_GPU":False,"h140_blind_calls":0}
    (owner/"private-h164/H164_DATA_PACKAGE_RECEIPT.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(build(args.owner),indent=2,sort_keys=True))
