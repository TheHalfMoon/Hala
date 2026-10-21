"""H52 original browser task factory: independent bytes/split/action audit.

No model outputs, hidden benchmark answers, user credentials or network IO.
"""
import hashlib, json
from pathlib import Path
from h52.factory import VERSION, RIGHTS, plans, render, canon, digest
from training.artifacts import validate_action_completion

def audit(root):
    root=Path(root)
    expected=plans()
    if (root/"plans.json").read_bytes()!=canon(expected):
        raise ValueError("Original authored task generator has changed")
    src=json.loads((root/"source.json").read_text("utf-8"))
    if src["schema"]!=VERSION or src["count"]!=96 or src["training_only"] is not True:
        raise ValueError("Source is not original train-only")
    if (src["authored_plans_sha256"]!=digest((root/"plans.json").read_bytes()) or
        src["original_html_sha256"]!=digest(canon(src["scenes"]))):
        raise ValueError("Original HTML/author receipts broken")
    for plan,record in zip(expected,src["scenes"]):
        if plan["id"]!=record["id"] or plan["rights"]!=RIGHTS:
            raise ValueError("Original plan identity")
        if (root/"html"/(plan["id"]+".html")).read_bytes()!=render(plan):
            raise ValueError("Original HTML changed")
        if record["sha256"]!=digest(render(plan)):
            raise ValueError("Original HTML digest changed")
    manifest=json.loads((root/"sft_manifest.json").read_text("utf-8"))
    capture=json.loads((root/"capture.json").read_text("utf-8"))
    rows=[json.loads(x) for x in (root/"authoring_sft.jsonl").read_text("utf-8").splitlines()]
    if (manifest["source_schema"]!=VERSION or manifest["source_sha256"]!=digest((root/"source.json").read_bytes()) or
        manifest["dataset_sha256"]!=digest((root/"authoring_sft.jsonl").read_bytes()) or
        manifest["source_kind"]!="synthetic" or manifest["split"]!="train" or
        manifest["rights_cleared"] is not True or manifest["media"] is None or
        manifest["independent_blind_test"] is not False or
        manifest["model_training_executed"] is not False or
        manifest["dev_or_benchmark_examples"]!=0 or len(manifest["media"])!=288):
        raise ValueError("Training-only manifest incorrect")
    if (capture["schema"]!="hala.h52.real-edge-three-step-replay.v1" or
        capture["source_sha256"]!=manifest["source_sha256"] or
        capture["successful_real_step_clicks"]!=288 or
        capture["all_completed_workflows"]!=96 or
        capture["wrong_clicks_tested"]!=4 or
        capture["real_service_access"] is not False or
        capture["model_inference"] is not False or
        len(capture["cases"])!=288 or len(rows)!=288):
        raise ValueError("Original Edge replay incomplete")
    mapping={r["path"]:r for r in manifest["media"]}
    if len(mapping)!=288 or len({x["sha256"] for x in mapping.values()})!=288:
        raise ValueError("Repeated or duplicate screenshot")
    languages={"ar":0,"en":0}
    for n,(row,record) in enumerate(zip(rows,capture["cases"])):
        plan=expected[n//3]
        stage=n%3
        screen=plan["id"]+"-s"+str(stage)
        photo="png/"+screen+".png"
        target=plan["targets"][stage]
        if (row["id"]!=screen or row["image_path"]!=photo or record["screen"]!=screen or
            record["source_id"]!=plan["id"] or record["stage"]!=stage or
            record["target_key"]!=target or row["split"]!="train" or
            row["family"]!=plan["family"] or row["language"]!=plan["lang"] or
            record["observed_step_after"]!=stage+1 or
            record["original_html_sha256"]!=digest(render(plan))):
            raise ValueError("Source/action/capture mismatch")
        languages[row["language"]]+=1
        image=(root/photo).read_bytes()
        if (not image.startswith(b"\x89PNG\r\n\x1a\n") or
            row["image_sha256"]!=digest(image) or
            mapping[photo]["sha256"]!=digest(image) or
            record["screenshot_sha256"]!=digest(image)):
            raise ValueError("Image source byte mismatch")
        completion=validate_action_completion(row["completion"])
        point=completion["point"]
        if point!=record["point"]:
            raise ValueError("Wrong pixel ground truth")
        matches=[c["key"] for c in record["controls"] if
            c["rect"][0]<=point[0]<c["rect"][0]+c["rect"][2] and
            c["rect"][1]<=point[1]<c["rect"][1]+c["rect"][3]]
        if matches!=[target] or plan["labels"][target] not in row["prompt"]:
            raise ValueError("Grounding geometry or instruction wrong")
    if languages!={"ar":144,"en":144}:raise ValueError("Bilingual imbalance")
    return dict(rows=len(rows),original_workflows=len(expected),
                independent_screenshots=len(mapping),language_rows=languages,
                dataset_sha256=manifest["dataset_sha256"],
                browser_wrong_targets_rejected=capture["wrong_clicks_tested"])

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="h52/assets")
    print(json.dumps(audit(p.parse_args().root),sort_keys=True))
