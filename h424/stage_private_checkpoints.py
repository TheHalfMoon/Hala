"""Stage SHA-verified H419 Kaggle model checkpoints in a PRIVATE dataset only.

No public model weights, screenshot pixels, source archives or secrets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from zipfile import ZipFile, ZIP_STORED

RECEIPT_SHA = "f8c491d8522a8349974d05318fded7cbe0349478be27efcbb118a8efdcfa37ac"
FINAL_SHA = "96d07b07cca54d15e9ce11da4d26fa3748564f09a2974d9a0b48e419ab03eb8c"
ADAPTER_BYTES = 12878000
STEPS = (1500, 3000, 4500, 6000, 7500, 9000)
ARCHIVE = "h424_h419_owner_private_checkpoint_bundle.zip"
MANIFEST = "h424_checkpoint_manifest.json"

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()

def verify_progress(text: str) -> None:
    progress = re.findall(
        r"H419_REAL_ORIGINAL_30000_NATIVE_GRADIENT\s+(\d+)\s+/\s+9452\s+"
        r"image_conditioned_samples_seen\s+(\d+)\s+loss\s+([0-9.]+)", text
    )
    if [(int(n), int(seen)) for n,seen,_ in progress] != [
        (step, step*4) for step in range(150, 9451, 150)
    ]:
        raise ValueError("Original H419 progress missing/duplicated/out of order")
    if any(not 0 <= float(loss) < 100 for _,_,loss in progress):
        raise ValueError("H419 provider progress loss invalid")

def verify_adapter(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != ADAPTER_BYTES:
        raise ValueError("Original H419 checkpoint bytes missing/wrong length")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not (64 < header_size < 2000000):
            raise ValueError("Original H419 safetensors header invalid")
        metadata = json.loads(stream.read(header_size))
    rows = [item for name,item in metadata.items() if name != "__metadata__"]
    if not 100 <= len(rows) <= 1000:
        raise ValueError("Original H419 LoRA tensor count invalid")
    for item in rows:
        offsets = item.get("data_offsets")
        if (item.get("dtype") not in ("F16","BF16","F32")
            or not isinstance(offsets,list) or len(offsets)!=2
            or not all(type(v) is int for v in offsets)
            or not 0 <= offsets[0] < offsets[1] <= ADAPTER_BYTES-8-header_size):
            raise ValueError("Original H419 tensor metadata invalid")

def stage(src: Path, log: Path, dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Create-only staging: output already populated")
    raw=(src/"receipt.json").read_bytes()
    if sha_bytes(raw)!=RECEIPT_SHA or (src/"receipt.sha256").read_text().strip()!=RECEIPT_SHA:
        raise ValueError("Original H419 owner provider receipt SHA changed")
    data=json.loads(raw)
    if (data.get("schema")!="hala.h419.private-new-30000-original-native-drag.v1"
        or data.get("status")!="COMPLETE_PRIVATE_30000_NEW_ORIGINAL_NATIVE_DRAG"
        or data.get("actual_optimizer_steps")!=9452
        or data.get("processed_training_examples")!=37808
        or data.get("adapter_sha256")!=FINAL_SHA
        or data.get("adapter_bytes")!=ADAPTER_BYTES
        or data.get("source_only_research_gate") is not False
        or data.get("model_release_qualified") is not False):
        raise ValueError("H419 original actual-model and failed research-gate evidence changed")
    log_raw=log.read_bytes()
    verify_progress(log_raw.decode("utf8"))
    folders={"final":src/"private-30k-original-native-drag-adapter",
             **{str(step):src/"private-checkpoints"/f"step-{step}" for step in STEPS}}
    snapshots={}
    contents={"receipt.json":src/"receipt.json", "receipt.sha256":src/"receipt.sha256"}
    for label,folder in folders.items():
        model=folder/"adapter_model.safetensors"
        config=folder/"adapter_config.json"
        verify_adapter(model)
        if label=="final" and sha_file(model)!=FINAL_SHA:
            raise ValueError("Final H419 adapter file does not match original receipt")
        conf=json.loads(config.read_text("utf8"))
        if conf.get("base_model_name_or_path") not in (None,"Qwen/Qwen3-VL-2B-Instruct"):
            raise ValueError("Original H419 PEFT model base drift")
        snapshots[label]={"step":9452 if label=="final" else int(label),
          "adapter_sha256":sha_file(model),"config_sha256":sha_file(config),
          "adapter_bytes":model.stat().st_size}
        contents[f"models/{label}/adapter_model.safetensors"]=model
        contents[f"models/{label}/adapter_config.json"]=config
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(dest/ARCHIVE,"w",ZIP_STORED) as out:
        for arc,path in contents.items():
            out.write(path,arcname=arc)
    manifest={
      "schema":"hala.h424.actual-owner-h419-checkpoints.v1",
      "owner_h419_receipt_sha256":RECEIPT_SHA,
      "owner_h419_provider_log_sha256":sha_bytes(log_raw),
      "archive_sha256":sha_file(dest/ARCHIVE),
      "model_release_qualified":False,
      "independent_external_ood":False,
      "original_completed_optimizer_steps":9452,
      "original_model_research_gate":False,
      "models":snapshots}
    manifest_raw=json.dumps(manifest,sort_keys=True,indent=2).encode()
    (dest/MANIFEST).write_bytes(manifest_raw)
    (dest/"dataset-metadata.json").write_text(json.dumps({
      "title":"Hala H424 Original H419 Model Checkpoints (Private)",
      "id":"abdulazizshehri/hala-h424-original-h419-checkpoints-private",
      "subtitle":"Actual source-authenticated H419 intermediate adapters, no release",
      "description":"PRIVATE 6 authentic H419 intermediate LoRA checkpoints plus final adapter, SHA-bound to owner Kaggle H419 provider. Final 9452-step model research-gate FAIL due old DEV regression. No raw screenshots, PHI, clinical autonomy or public weights.",
      "isPrivate":True,
      "licenses":[{"name":"apache-2.0"}]},sort_keys=True,indent=2))
    return {"manifest_sha256":sha_bytes(manifest_raw),
            "archive_sha256":manifest["archive_sha256"],
            "checkpoint_shas":{k:v["adapter_sha256"] for k,v in snapshots.items()}}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--provider-log",type=Path,required=True)
    p.add_argument("--dest",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(stage(args.source,args.provider_log,args.dest),sort_keys=True))
