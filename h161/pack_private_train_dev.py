"""H161 owner-private TRAIN labels and DEV without gold, create-only Kaggle pack."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from h161.audit_source import audit
from h161.original_apps_source import SCHEMA

MANIFEST_SHA="0710797074d8407640fa8c0e65bbfe8672175a644483701ce156aaab02621997"
DATASET="abdulazizshehri/hala-h161-private-two-original-app-sft-v1"
MODEL_ARCHIVE="h157-frozen-research-lora.zip"
MODEL_ARCHIVE_SHA="14e3a9a61123a92fb23d53297531831fa19ff7d2435d6704717af7a90c08e808"
H157_CHECKPOINT_SHA="181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8"
PUBLIC_FIELDS={"split","app","task","language","index","image","image_sha256","instruction"}


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def prepare(owner):
    owner=Path(owner)
    aud=audit(owner)
    if aud["source_manifest_sha256"]!=MANIFEST_SHA:
        raise ValueError("H161 source manifest revision not preregistered")
    qa=json.loads((owner/"private-h161"/"H161_INDEPENDENT_PIXEL_QA.json").read_text("utf8"))
    if qa.get("status")!="H161_72_OF_72_CROSS_CONTEXT_EXACT_NATIVE_SCREENSHOT_MATCH":
        raise ValueError("H161 exact source/replay image identity not proven")
    src=owner/"private-h161"/"two-real-app-source-v1"
    manifest=json.loads((src/"manifest.json").read_text("utf8"))
    if manifest["schema"]!=SCHEMA or len(manifest["private_cases"])!=72:
        raise ValueError("H161 source schema/total changed")
    root=owner/"private-h161"/"kaggle-sft-v1"
    if root.exists():
        raise FileExistsError("H161 training/DEV split archive is create-once")
    labels={}
    for split in ("TRAIN","DEV"):
        rows=[]
        for record in manifest["private_cases"]:
            if record["split"]!=split:
                continue
            metadata={k:record[k] for k in PUBLIC_FIELDS}
            if split=="TRAIN":
                # Gold only on TRAIN archive, strictly no DEV gold on Kaggle.
                metadata["training_supervision"]=record["training_supervision"]
            rows.append(metadata)
        labels[split]=rows
    if (len(labels["TRAIN"])!=48 or len(labels["DEV"])!=24 or
        any("training_supervision" in r for r in labels["DEV"])):
        raise ValueError("H161 DEV gold leaked into GPU dataset")
    if ({r["image_sha256"] for r in labels["TRAIN"]} &
        {r["image_sha256"] for r in labels["DEV"]}):
        raise ValueError("H161 train and eval original scenes overlap")

    model=(owner/"private-h158"/"kaggle-eval-only-v1"/MODEL_ARCHIVE)
    if sha(model.read_bytes())!=MODEL_ARCHIVE_SHA:
        raise ValueError("H161 H157 original research starting LoRA ZIP changed")
    with ZipFile(model) as z:
        if sha(z.read("adapter_model.safetensors"))!=H157_CHECKPOINT_SHA:
            raise ValueError("H161 original H157 adapter SHA unexpectedly changed")
    root.mkdir(parents=True,exist_ok=False)
    archives={}
    for split,items in labels.items():
        basename=f"h161-{split.lower()}-original-app-source.zip"
        filename=root/basename
        export={"schema":"hala.h161.freeze-private-"+split.lower()+"-source.v1",
                "parent_manifest_sha256":MANIFEST_SHA,
                "split":split,"case_count":len(items),"private_cases":items,
                "DEV_gold_included":False,"h140_blind_calls":0,
                "paid_gpu":False}
        with ZipFile(filename,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
            z.writestr("manifest.json",json.dumps(export,ensure_ascii=False,
                       indent=2,sort_keys=True)+"\n")
            for item in items:
                z.write(src/"images"/item["image"],"images/"+item["image"])
        with ZipFile(filename) as z:
            if (z.testzip() is not None or
                set(z.namelist())!={"manifest.json"}|
                {"images/"+item["image"] for item in items}):
                raise ValueError("H161 private training/DEV archive malformed")
        archives[split]={"filename":basename,
                          "sha256":sha(filename.read_bytes()),
                          "manifest_sha256":sha(ZipFile(filename).read("manifest.json"))}
    shutil.copyfile(model,root/MODEL_ARCHIVE)
    (root/"dataset-metadata.json").write_text(json.dumps({
        "id":DATASET,"title":"hala-h161-private-two-original-app-sft-v1",
        "licenses":[{"name":"other"}]},indent=2)+"\n","utf8")
    receipt={
        "status":"H161_TRAIN_48_DEV_24_NO_DEV_GOLD_PRIVATE_KAGGLE_READY",
        "source_manifest_sha256":MANIFEST_SHA,
        "archives":archives,"original_app_count":2,
        "train":48,"dev":24,
        "starting_model_weights_sha256":H157_CHECKPOINT_SHA,
        "optimizer_steps":0,"paid_gpu":False,
        "h140_blind_calls":0,"Hala35B_unchanged":True,
        "DEV_labels_uploaded":False,
    }
    (owner/"private-h161"/"H161_KAGGLE_PACK_RECEIPT.json").write_text(
        json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner",required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.owner),ensure_ascii=False,indent=2,sort_keys=True))
