"""H157 owner PRIVATE free-Kaggle source and frozen H156 adapter packer.

Never publish images, owner case labels, or updated LoRA. Fail closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from h157.audit_spatial_source import audit

ADAPTER_SHA="732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6"
DATASET="abdulazizshehri/hala-h157-private-spatial-action-v1"
MODEL_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package(owner):
    owner=Path(owner)
    receipt=audit(owner)
    source=owner/"private-h157"/"source-spatial-train-dev-v1"
    model=(owner/"private-h156"/"kaggle-output-v4"/
           "hala-h156-typed-action-lora"/"private-h156-typed-action-lora")
    path=model/"adapter_model.safetensors"
    if (not path.is_file() or sha(path.read_bytes())!=ADAPTER_SHA or
        {p.name for p in model.iterdir() if p.is_file()}!=MODEL_FILES):
        raise ValueError("H157 requires exact new trained H156 research model")
    root=owner/"private-h157"/"kaggle-source-v1"
    if root.exists():
        raise FileExistsError("H157 Kaggle private dataset packaging create-only")
    root.mkdir(parents=True,exist_ok=False)
    dataset_zip=root/"h157-original-source.zip"
    adapter_zip=root/"h156-trained-typed-action-lora.zip"
    cases=json.loads((source/"manifest.json").read_text("utf8"))["private_cases"]
    with ZipFile(dataset_zip,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        z.write(source/"manifest.json","manifest.json")
        for case in cases:
            name=case["image"]
            data=(source/"images"/name).read_bytes()
            if sha(data)!=case["image_sha256"]:
                raise ValueError("H157 source changed after audit")
            z.writestr("images/"+name,data)
    with ZipFile(dataset_zip) as z:
        if (z.testzip() or len(z.namelist())!=len(cases)+1 or
            sha(z.read("manifest.json"))!=receipt["manifest_sha256"]):
            raise ValueError("H157 private source ZIP checksum mismatch")
    with ZipFile(adapter_zip,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for name in sorted(MODEL_FILES):
            z.write(model/name,name)
    with ZipFile(adapter_zip) as z:
        if (z.testzip() or set(z.namelist())!=MODEL_FILES or
            sha(z.read("adapter_model.safetensors"))!=ADAPTER_SHA):
            raise ValueError("H157 new research model ZIP changed")
    (root/"dataset-metadata.json").write_text(json.dumps({
        "id":DATASET,"title":"hala-h157-private-spatial-action-v1",
        "licenses":[{"name":"other"}],
    },indent=2)+"\n",encoding="utf8")
    return {
        "status":"H157_PRIVATE_SOURCE_PACKAGE_READY",
        "dataset":DATASET,
        "train":receipt["train"],"dev":receipt["dev"],
        "original_manifest_sha256":receipt["manifest_sha256"],
        "source_zip_sha256":sha(dataset_zip.read_bytes()),
        "model_adapter_sha256":ADAPTER_SHA,
        "model_zip_sha256":sha(adapter_zip.read_bytes()),
        "private_only":True,"paid_gpu":False,
        "h140_blind_calls":0,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    args=parser.parse_args()
    print(json.dumps(package(args.owner_root),sort_keys=True,indent=2))
