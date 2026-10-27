"""Freeze H163 new original app source, no oracle positions exported to Kaggle."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from h163.source import GIT_SHA, SCHEMA

SOURCE_SHA="73f2416e65c3a843465d1dc2cce5614b91b050708b8e6d6388ba77add785423a"
OLDER_MODEL_SHA="181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8"
TRAINED_MODEL_SHA="72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c"
OLD_ZIP_SHA="14e3a9a61123a92fb23d53297531831fa19ff7d2435d6704717af7a90c08e808"
NEW_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"
DATASET="abdulazizshehri/hala-h163-private-new-gridstack-ood-v1"
ORIGINAL_ZIP="h163-new-original-gridstack-scenes.zip"
OLDER_ZIP="h157-original-frozen-research-lora.zip"
TRAINED_ZIP="h161-original-trained-research-lora.zip"
FIELDS=("index","app","task","language","layout",
        "pre_column","target_column","image","image_sha256","instruction")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def prepare(owner):
    owner=Path(owner)
    original=owner/"private-h163"/"source-once-v1"
    raw=(original/"manifest.json").read_bytes()
    proof=json.loads((owner/"private-h163"/"H163_SOURCE_PIXEL_QA.json").read_text("utf8"))
    if (digest(raw)!=SOURCE_SHA or
        proof.get("status")!="H163_16_OF_16_ORIGINAL_NEW_APP_EXACT_PIXEL_REPLAY_PASS"):
        raise ValueError("H163 no reproducible locked new original app 16/16 source")
    doc=json.loads(raw)
    source_rows=doc.get("private_cases")
    if (doc.get("schema")!=SCHEMA or doc.get("upstream_git_sha")!=GIT_SHA
        or len(source_rows)!=16 or doc.get("native_reference_success")!=16
        or doc.get("heldout_evaluation_only") is not True):
        raise ValueError("H163 original MIT source/benchmark changed")
    minimal=[]
    for item in source_rows:
        if "owner_gold_click" not in item:
            raise ValueError("H163 original private gold missing for later owner scoring")
        minimal.append({field:item[field] for field in FIELDS})
    if len({x["image_sha256"] for x in minimal})!=16:
        raise ValueError("H163 duplicate original screenshots")
    older=owner/"private-h162"/"kaggle-eval-v1"/"h157-starting-research-lora.zip"
    trained=owner/"private-h162"/"kaggle-eval-v1"/"h161-mixed-trained-research-lora.zip"
    if digest(older.read_bytes())!=OLD_ZIP_SHA or digest(trained.read_bytes())!=NEW_ZIP_SHA:
        raise ValueError("H163 frozen H157/H161 research LoRA ZIP checksum drift")
    with ZipFile(older) as z:
        if digest(z.read("adapter_model.safetensors"))!=OLDER_MODEL_SHA:
            raise ValueError("H163 baseline research LoRA changed")
    with ZipFile(trained) as z:
        if digest(z.read("adapter_model.safetensors"))!=TRAINED_MODEL_SHA:
            raise ValueError("H163 H161 trained research LoRA changed")
    dest=owner/"private-h163"/"kaggle-eval-v1"
    if dest.exists():
        raise FileExistsError("H163 external new-app model inputs create-only")
    dest.mkdir(parents=True,exist_ok=False)
    export={"schema":"hala.h163.new-original-gridstack-inference-only.v1",
            "source_manifest_sha256":SOURCE_SHA,"original_app_sha":GIT_SHA,
            "cases":16,"optimizer_steps":0,"h140_blind_calls":0,
            "reference_points_uploaded":False,"private_cases":minimal}
    with ZipFile(dest/ORIGINAL_ZIP,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr("manifest.json",json.dumps(export,ensure_ascii=False,
                   indent=2,sort_keys=True)+"\n")
        for item in minimal:
            blob=(original/"images"/item["image"]).read_bytes()
            if digest(blob)!=item["image_sha256"]:
                raise ValueError("H163 original image content SHA modified")
            z.writestr("images/"+item["image"],blob)
    with ZipFile(dest/ORIGINAL_ZIP) as z:
        if (len(z.namelist())!=17 or z.testzip() is not None or
            b"owner_gold_click" in z.read("manifest.json")):
            raise ValueError("H163 leaked gold original target or incomplete")
    shutil.copyfile(older,dest/OLDER_ZIP)
    shutil.copyfile(trained,dest/TRAINED_ZIP)
    (dest/"dataset-metadata.json").write_text(json.dumps({
        "id":DATASET,"title":DATASET.split("/")[-1],
        "licenses":[{"name":"other"}]},indent=2)+"\n","utf8")
    receipt={"status":"H163_NEW_ORIGINAL_APP_GPU_INPUT_FROZEN",
             "source_manifest_sha256":SOURCE_SHA,
             "source_zip_sha256":digest((dest/ORIGINAL_ZIP).read_bytes()),
             "source_zip_internal_manifest_sha256":
               digest(ZipFile(dest/ORIGINAL_ZIP).read("manifest.json")),
             "before_adapter_sha256":OLDER_MODEL_SHA,
             "after_adapter_sha256":TRAINED_MODEL_SHA,
             "cases":16,"model_calls":0,"optimizer_steps":0,
             "h140_blind_calls":0,"paid_GPU":False,"gold_uploaded":False}
    (owner/"private-h163"/"H163_PACK_FREEZE_RECEIPT.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.owner),indent=2,sort_keys=True))
