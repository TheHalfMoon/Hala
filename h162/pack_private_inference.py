"""H162 third ORIGINAL app private frozen inference-only OOD source pack.

Only 16 source images/instructions, NO owner gold action coordinates on GPU.
Freeze H157 and H161 2B rank8 LoRAs. Free Kaggle T4 inference-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile
import shutil

from h162.original_third_app import SCHEMA

APP_SHA="83413e1f5fab4690bb46b23201a3438a427ef265"
SOURCE_SHA="1f8858e1d39c54a8f9a748f4a0a8417316d9c113780f5a7963ffd8bfc54d4089"
BASE_SHA="181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8"
TRAINED_SHA="72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c"
BASE_ZIP_SHA="14e3a9a61123a92fb23d53297531831fa19ff7d2435d6704717af7a90c08e808"
DATASET="abdulazizshehri/hala-h162-private-third-original-app-v1"
SOURCE_ZIP="h162-third-html5-original-source.zip"
OLD_ZIP="h157-starting-research-lora.zip"
NEW_ZIP="h161-mixed-trained-research-lora.zip"
MODEL_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(owner):
    owner=Path(owner)
    source=owner/"private-h162"/"original-html5-board-source-once-v1"
    manifest_bytes=(source/"manifest.json").read_bytes()
    original=json.loads(manifest_bytes)
    proof=json.loads((owner/"private-h162"/"H162_SOURCE_PIXEL_QA.json")
                     .read_text("utf8"))
    if (sha(manifest_bytes)!=SOURCE_SHA or
        original.get("schema")!=SCHEMA or
        original.get("upstream_sha")!=APP_SHA or
        original.get("source_task_count")!=16 or
        len(original["private_cases"])!=16 or
        proof.get("status")!="H162_16_EXACT_ORIGINAL_THIRD_APP_REPLAY_PASS"):
        raise ValueError("H162 independent new app source frozen SHA invalid")
    root=owner/"private-h162"/"kaggle-eval-v1"
    if root.exists():
        raise FileExistsError("H162 owner PRIVATE dataset create-once")
    cases=[]
    for row in original["private_cases"]:
        cases.append({k:row[k] for k in (
            "index","task","language","app","image","image_sha256","instruction")})
    if len({r["image_sha256"] for r in cases})!=16:
        raise ValueError("H162 third app original images duplicated")
    document={
        "schema":"hala.h162.original-html5-kanban-external-app-model-input.v1",
        "source_app_sha":APP_SHA,"source_manifest_sha256":SOURCE_SHA,
        "source_cases":16,"model_optimizer_steps":0,
        "historically_seen_app_NOT_sealed_BLIND":True,
        "private_cases":cases,
    }
    current=owner/"private-h161"/"kaggle-output-v1"/"hala-h161-multi-original-app-trained-model"
    trained_report=json.loads((current/"h161-genuine-learning-receipt.json")
                              .read_text("utf8"))
    if (trained_report.get("status")!=
        "H161_GENUINE_48_T4_TWO_REAL_APP_TRAIN_STEPS_COMPLETE" or
        trained_report.get("new_mixed_app_adapter_sha256")!=TRAINED_SHA or
        trained_report.get("optimizer_steps")!=48 or
        trained_report.get("dev_labels_seen_by_GPU") is not False):
        raise ValueError("H162 starting trained checkpoint not verified")
    base=owner/"private-h161"/"kaggle-sft-v1"/"h157-frozen-research-lora.zip"
    if sha(base.read_bytes())!=BASE_ZIP_SHA:
        raise ValueError("H162 original base LoRA ZIP mutated")
    learned=current/"private-h161-mixed-original-app-research-lora"
    if ({p.name for p in learned.iterdir() if p.is_file()}!=MODEL_FILES or
        sha((learned/"adapter_model.safetensors").read_bytes())!=TRAINED_SHA):
        raise ValueError("H162 trained model full identity not admitted")
    root.mkdir(parents=True,exist_ok=False)
    with ZipFile(root/SOURCE_ZIP,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr("manifest.json",json.dumps(
            document,sort_keys=True,ensure_ascii=False,indent=2)+"\n")
        for item in cases:
            png=(source/"images"/item["image"]).read_bytes()
            if sha(png)!=item["image_sha256"]:
                raise ValueError("H162 original app private image SHA changed")
            z.writestr("images/"+item["image"],png)
    shutil.copyfile(base,root/OLD_ZIP)
    with ZipFile(root/NEW_ZIP,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for name in sorted(MODEL_FILES):
            z.write(learned/name,name)
    with ZipFile(root/SOURCE_ZIP) as z:
        if (z.testzip() is not None or len(z.namelist())!=17 or
            "owner_reference_action" in z.read("manifest.json").decode()):
            raise ValueError("H162 private source leaked gold action or missing media")
    (root/"dataset-metadata.json").write_text(json.dumps({
        "id":DATASET,"title":"hala-h162-private-third-original-app-v1",
        "licenses":[{"name":"other"}]},indent=2)+"\n","utf8")
    return {
        "status":"H162_EXPLORATORY_ORIGINAL_THIRD_APP_GPU_INPUT_READY",
        "source_zip_sha256":sha((root/SOURCE_ZIP).read_bytes()),
        "new_lora_zip_sha256":sha((root/NEW_ZIP).read_bytes()),
        "old_lora_sha256":BASE_SHA,"new_lora_sha256":TRAINED_SHA,
        "model_calls":0,"training_steps":0,"paid_GPU":False,
        "h140_blind_calls":0,"source_cases":16,
        "gold_points_uploaded":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner),indent=2,sort_keys=True))
