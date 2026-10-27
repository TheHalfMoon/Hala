"""H158 owner-private 16 original TodoMVC OOD source + TWO frozen adapters.

Evaluation ONLY. No optimizer data, no training, no model publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from h158.audit_source import audit

DATASET="abdulazizshehri/hala-h158-private-second-app-click-v1"
SHA_H156="732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6"
SHA_H157="181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8"
FILES={"README.md","adapter_config.json","adapter_model.safetensors"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(owner):
    owner=Path(owner)
    receipt=audit(owner)
    root=owner/"private-h158"/"kaggle-eval-only-v1"
    if root.exists():
        raise FileExistsError("H158 independent app evaluation create-once")
    old=(owner/"private-h156"/"kaggle-output-v4"/"hala-h156-typed-action-lora"/
         "private-h156-typed-action-lora")
    new=(owner/"private-h157"/"kaggle-output-v1"/"hala-h157-spatial-action-lora"/
         "private-h157-spatial-research-lora")
    for src,checksum in ((old,SHA_H156),(new,SHA_H157)):
        if ({f.name for f in src.iterdir() if f.is_file()}!=FILES or
            sha((src/"adapter_model.safetensors").read_bytes())!=checksum):
            raise ValueError("H158 frozen source LoRA checksum/file list wrong")
    root.mkdir(parents=True,exist_ok=False)
    images=owner/"private-h158"/"todomvc-source-once-v1"
    manifest=json.loads((images/"manifest.json").read_text("utf8"))
    source_zip=root/"h158-original-todomvc-source.zip"
    with ZipFile(source_zip,"x",compression=ZIP_DEFLATED,compresslevel=3) as zipfile:
        zipfile.write(images/"manifest.json","manifest.json")
        for case in manifest["private_cases"]:
            data=(images/"images"/case["image"]).read_bytes()
            if sha(data)!=case["image_sha256"]:
                raise ValueError("H158 owner original PNG changed before upload")
            zipfile.writestr("images/"+case["image"],data)
    if sha(manifest_bytes:=(images/"manifest.json").read_bytes())!=receipt["manifest_sha256"]:
        raise ValueError("H158 frozen original Todo source SHA mismatch")
    with ZipFile(source_zip) as z:
        if z.testzip() or len(z.namelist())!=17 or sha(z.read("manifest.json"))!=receipt["manifest_sha256"]:
            raise ValueError("H158 source ZIP format not reproducible")
    for role,folder in (("h156",old),("h157",new)):
        with ZipFile(root/(f"{role}-frozen-research-lora.zip"),"x",
                     compression=ZIP_DEFLATED,compresslevel=3) as zipped:
            for name in sorted(FILES):
                zipped.write(folder/name,name)
    (root/"dataset-metadata.json").write_text(json.dumps({
        "id":DATASET,"title":"hala-h158-private-second-app-click-v1",
        "licenses":[{"name":"other"}]
    },indent=2)+"\n",encoding="utf8")
    return {"status":"H158_PRIVATE_TWO_MODEL_OOD_INFERENCE_PACKAGE_READY",
            "private_dataset":DATASET,
            "independent_app_images":16,
            "original_manifest_sha256":receipt["manifest_sha256"],
            "source_zip_sha256":sha(source_zip.read_bytes()),
            "old_lora_sha256":SHA_H156,"new_lora_sha256":SHA_H157,
            "model_training_or_optimizer":False,
            "paid_compute":False,"h140_blind_calls":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root),sort_keys=True,indent=2))
