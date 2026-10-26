"""Export a PRIVATE inference-only copy of H139 independent blind images.

No bboxes, source task ids, application state or gold gestures ever travel
with the inference input. This is separate from the TRAIN/DEV Kaggle kernel.
"""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from h139.audit import qualify as audit_sources
from h139.registry import UPSTREAM,check_split_manifest,sha

SCHEMA="hala.h139.private-redacted-blind8-only-model-input.v1"
FIELDS=("id","app","app_git_sha","lang","instruction","image","image_sha256")
FORBIDDEN=("start_bbox","end_bbox","source_task_id","seed_index","reference_coordinates")

def export_private(source_root,output,old_manifests=()):
    source_root=Path(source_root).resolve()
    output=Path(output).resolve()
    if source_root.name!="private-h139" or source_root not in output.parents:
        raise ValueError("H139 inference source/archive must remain owner-private")
    if output.exists():
        raise ValueError("H139 blind export is create-only; never overwrite")
    audit_sources(source_root,old_manifests)
    folder=source_root/"blind8"
    original=(folder/"manifest.json").read_bytes()
    doc=json.loads(original)
    cases=check_split_manifest(doc)
    if doc.get("split")!="blind_test" or len(cases)!=8:
        raise ValueError("H139 blind source must be exactly eight immutable tasks")
    redacted=[]
    seen=set()
    for row in cases:
        if row.get("app")!="ood-board" or row.get("role")!="blind_test":
            raise ValueError("H139 forbid TRAIN or DEV in independent source")
        if any(f not in row for f in FIELDS):
            raise ValueError("H139 missing redacted model input field")
        img=folder/row["image"]
        if img.parent!=folder or not img.is_file():
            raise ValueError("H139 unexpected blind image path")
        if row["image"] in seen or sha(img.read_bytes())!=row["image_sha256"]:
            raise ValueError("H139 duplicate / tampered private screenshot")
        seen.add(row["image"])
        redacted.append({key:row[key] for key in FIELDS})
    if {x["lang"] for x in redacted}!={"ar","en"} or sum(
            x["lang"]=="ar" for x in redacted)!=4:
        raise ValueError("H139 must preserve balanced bilingual OOD split")
    safe={"schema":SCHEMA,"private":True,"source_role":"blind_test",
          "training_eligible":False,"original_manifest_sha256":sha(original),
          "source_app_sha":UPSTREAM["ood-board"]["git_sha"],
          "case_count":8,"cases":redacted}
    data=(json.dumps(safe,sort_keys=True,indent=2)+"\n").encode("utf8")
    if any(key.encode() in data for key in FORBIDDEN):
        raise ValueError("H139 source labels leaked to inference transport")
    output.parent.mkdir(parents=True,exist_ok=True)
    # Fail closed: an invalid partial archive must never be accepted.
    temp=output.with_suffix(".incomplete")
    if temp.exists():
        raise ValueError("H139 prior partial archive needs owner review")
    try:
        with ZipFile(temp,"w",compression=ZIP_DEFLATED) as z:
            z.writestr("manifest.json",data)
            for row in redacted:
                z.write(folder/row["image"],arcname=row["image"])
        with ZipFile(temp) as z:
            if set(z.namelist())!={"manifest.json"}|seen:
                raise ValueError("H139 redacted ZIP content mismatch")
            if z.read("manifest.json")!=data:
                raise ValueError("H139 redacted manifest changed in transit")
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()
    return {"export_sha256":sha(output.read_bytes()),"manifest_sha256":sha(data),
            "source_manifest_sha256":sha(original),"images":len(redacted),
            "truth_labels_exported":False}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--old-manifests",nargs="*",default=[])
    args=p.parse_args()
    print(json.dumps(export_private(args.private_root,args.output,args.old_manifests),
                     sort_keys=True))
