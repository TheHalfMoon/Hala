"""H421 author-visible same-screen/two-instruction native-drag holdout.

This evaluates *instruction conditioning*: a native vision model must select
two materially different targets on the SAME unseen original screenshot,
based on two different source-original task instructions.

Source: only original H60 TRAIN labels and JPEGs in nine SHA-pinned
owner-private archives. Every H417 TRAIN, H417 DEV, H419 TRAIN and
H419 old/new DEV source-image SHA excluded. Underlying original source
task SHA also disjoint from any previously trained/evaluated H60 image.

Creates 64 untouched source images per family (slider, slide_resize,
text_highlight), two native actions each = 192 original unseen screens,
384 original annotation targets. This is still author-visible research
source holdout, NOT third-party sealed independent OOD.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import zipfile

from h419.kaggle import h419_native_drag_continuation as model
from h418.raw_original_corpus import h417_train_and_dev_sets
from h418.curate import canonical
from h420.curate_alternative_drags import native_tuple

DOMAINS=("slider","slide_resize","text_highlight")
PER_DOMAIN=64
TARGET_PAIRS=PER_DOMAIN*len(DOMAINS)
REQUIRED_ACTIONS=2*TARGET_PAIRS
MIN_DISTANCE=0.15
SEED="hala-h421-unused-original-two-task-dev-source"
OUT="h421_original_same_image_two_instructions.jsonl"
RECEIPT="h421_source_counterfactual_receipt.json"


def candidate_pair(rows):
    """Choose a source-original, meaningfully separated native action pair."""
    if type(rows) is not list or len(rows)<2:
        return None
    ranked=[]
    for row in rows:
        case=model.case_from_dragon(row)
        if row["ordered"] is not case["ordered"]:
            raise ValueError("H421 original native ordered flag differs")
        key=native_tuple(row["completion"])
        ranking=sha256((SEED+"|"+row["id"]).encode()).hexdigest()
        ranked.append((ranking,case,row,key))
    best=None
    for i,(_,a,raw_a,pa) in enumerate(ranked):
        for _,b,raw_b,pb in ranked[i+1:]:
            if (a["image_sha256"]!=b["image_sha256"] or
                a["domain"]!=b["domain"] or a["body"]==b["body"] or
                pa==pb):
                continue
            w,h=int(raw_a["image_width"]),int(raw_a["image_height"])
            if (w,h)!=(int(raw_b["image_width"]),int(raw_b["image_height"])):
                raise ValueError("H421 same original screenshot conflicting dimensions")
            shift=sum(abs(x-y) for before,after in zip(pa[:2],pb[:2])
                      for x,y in zip(before,after))
            ratio=shift/float(w+h)
            if ratio<MIN_DISTANCE:
                continue
            rank=(ratio,sha256((SEED+"|"+a["id"]+"|"+b["id"]).encode()).hexdigest())
            if best is None or rank>best[0]:
                best=(rank,a,b,raw_a,raw_b)
    return best


def original_sources(root):
    root=Path(root)
    paths={name:root/"h415"/name for name,_ in model.SOURCE_SHAS.values()}
    paths.update({name:root/"h418"/name for name in model.H419_EXTRA_SHAS})
    paths.update({
        model.BIL_FILE:root/"h415"/model.BIL_FILE,
        model.CURRIC_FILE:root/"h415"/model.CURRIC_FILE,
        model.H418_ROWS:root/"new-30k"/model.H418_ROWS,
        model.H418_RECEIPT_FILE:root/"new-30k"/model.H418_RECEIPT_FILE,
        model.PARENT_ADAPTER_FILE:root/"parent"/model.PARENT_ADAPTER_FILE,
        model.PARENT_RECEIPT_FILE:root/"parent"/model.PARENT_RECEIPT_FILE,
        "h417_original_receipt.sha256":root/"parent"/"h417_original_receipt.sha256",
    })
    if any(not p.is_file() for p in paths.values()):
        raise ValueError("H421 missing original private SHA-pinned source files")
    return paths


def audit(root):
    root=Path(root)
    old=model.only_input_file
    zips=None
    extras=[]
    try:
        paths=original_sources(root)
        model.only_input_file=lambda name:paths[name]
        zips,extras,train,dev,order,old_dev,new_dev=model.h419_prepare_sources()
        prior_train,prior_dev=h417_train_and_dev_sets(root/"h415")
        reserved=(set(prior_train)|set(prior_dev)|
                  {r["image_sha256"] for r in train+dev})
        if len(train)!=30976 or len(dev)!=240 or len(prior_train)!=11952 or len(prior_dev)!=144:
            raise ValueError("H421 original former train/dev source counts shifted")
        available={}
        blocked_tasks=set()
        action_rows=Counter()
        # First pass bans ALL original source-task IDs associated with any
        # H417/H419 train or DEV original image, regardless of action choice.
        for name,z in zips["_by_file"].items():
            manifest=model.strict_load(z.read("manifest.json").decode("utf8"))
            domain=manifest["source_domain"]
            with z.open("labels.jsonl") as source:
                for b in source:
                    if not b.strip():continue
                    rec=model.strict_load(b.decode("utf8"))
                    image_sha=rec["image_sha256"]
                    task=rec["source_task_sha256"]
                    action_rows[domain]+=1
                    if image_sha in reserved:
                        blocked_tasks.add(task)
                    if (image_sha not in reserved and domain in DOMAINS
                        and type(task) is str and len(task)==64):
                        family=available.setdefault(image_sha,{
                            "domain":domain,"zip":name,"rows":[],"tasks":set()})
                        if family["domain"]!=domain or family["zip"]!=name:
                            raise ValueError("H421 original candidate present in multiple source families")
                        # Limit memory on owner Windows; deterministic top16
                        # by source annotation ID keeps a broad task sample.
                        family["rows"].append(rec)
                        if len(family["rows"])>16:
                            family["rows"].sort(key=lambda r:sha256(
                                (SEED+"|"+r["id"]).encode()).digest())
                            del family["rows"][16:]
                        family["tasks"].add(task)
        if sum(action_rows.values())!=500554:
            raise ValueError("H421 exact original 500554 TRAIN actions absent")
        by_domain={d:[] for d in DOMAINS}
        dropped=Counter()
        for image_sha,entry in available.items():
            if entry["tasks"]&blocked_tasks:
                dropped["underlying_source_task_seen_by_parent_or_dev"]+=1
                continue
            pair=candidate_pair(entry["rows"])
            if pair is None:
                dropped["not_two_different_instruction_and_target"]+=1
                continue
            _,a,b,ra,rb=pair
            if ra["source_task_sha256"] in blocked_tasks or rb["source_task_sha256"] in blocked_tasks:
                raise ValueError("H421 source task signature already used")
            a.update(source_zip=entry["zip"],
                     image_width=ra["image_width"],image_height=ra["image_height"],
                     original_source_task_sha256=ra["source_task_sha256"])
            b.update(source_zip=entry["zip"],
                     image_width=rb["image_width"],image_height=rb["image_height"],
                     original_source_task_sha256=rb["source_task_sha256"])
            by_domain[entry["domain"]].append((image_sha,a,b))
        selected=[]
        used_source_tasks=set()
        for domain in DOMAINS:
            candidates=sorted(by_domain[domain],key=lambda x:sha256(
                (SEED+"|"+x[0]).encode()).hexdigest())
            for image_sha,a,b in candidates:
                ids={a["original_source_task_sha256"],
                     b["original_source_task_sha256"]}
                if ids&used_source_tasks:continue
                # A and B may share their own H60 underlying source-task
                # identity, but that identity must not appear in other
                # selected images or any training/evaluation source.
                used_source_tasks.update(ids)
                selected.append((image_sha,a,b))
                if sum(x[1]["domain"]==domain for x in selected)>=PER_DOMAIN:
                    break
        distribution=Counter(pair[1]["domain"] for pair in selected)
        if (len(selected)!=TARGET_PAIRS or
            distribution!={d:PER_DOMAIN for d in DOMAINS}):
            raise ValueError("H421 not enough untouched source-original counterfactual images")
        rows=[]
        ratios=[]
        for i,(sha,a,b) in enumerate(selected):
            if sha in reserved:
                raise ValueError("H421 protected TRAIN/DEV screenshot reused")
            # Exact source raw JPEG must be real, byte-identical and display
            # the same geometry under both original instructions.
            for case in (a,b):
                im,w,h=model.source_image(case,zips)
                if (w,h)!=(case["image_width"],case["image_height"]):
                    raise ValueError("H421 image and native action pixel geometry drift")
            pa=model.target(a,w,h)
            pb=model.target(b,w,h)
            shift=sum(abs(x-y) for before,after in zip(
                native_tuple({"type":"drag","start_point":a["start"],
                              "end_point":a["end"],"ordered":a["ordered"],
                              "coordinate_space":"source_image_pixels"})[:2],
                native_tuple({"type":"drag","start_point":b["start"],
                              "end_point":b["end"],"ordered":b["ordered"],
                              "coordinate_space":"source_image_pixels"})[:2])
                      for x,y in zip(before,after))
            ratio=shift/float(w+h)
            if ratio<MIN_DISTANCE:
                raise ValueError("H421 candidate source targets no longer distinct")
            ratios.append(round(ratio,6))
            for idx,case in enumerate((a,b)):
                rows.append({
                  "schema":"hala.h421.same-original-image-two-instructions.v1",
                  "pair_id":i,"alternative_index":idx,
                  "id":case["id"],"split":"dev",
                  "source_dataset":"Hcompany/DragOn",
                  "source_revision":model.SOURCE_REV,
                  "source_original_split":"train",
                  "source_domain":case["domain"],
                  "source_zip":case["source_zip"],
                  "source_zip_sha256":model.H419_ALL_SOURCE_SHAS[case["source_zip"]],
                  "original_source_task_sha256":case["original_source_task_sha256"],
                  "image_path":case["image_path"],
                  "image_sha256":case["image_sha256"],
                  "image_width":w,"image_height":h,
                  "language":"en","intent":case["body"],
                  "native_drag_target":pa if idx==0 else pb,
                  "same_image_pair_native_target_ratio":round(ratio,6),
                  "source_original_jpeg_bytes_sha_verified":True,
                  "external_independent_ood":False,
                })
            if (i+1)%32==0:
                print("H421_UNSEEN_SOURCE_COUNTERFACTUAL_PAIRS",i+1,"/",TARGET_PAIRS,
                      flush=True)
        if (len(rows)!=REQUIRED_ACTIONS or
            len({x["id"] for x in rows})!=REQUIRED_ACTIONS or
            len({x["image_sha256"] for x in rows})!=TARGET_PAIRS or
            len(set(used_source_tasks)&blocked_tasks)!=0):
            raise ValueError("H421 original source counterfactual DEV group incomplete")
        return rows,{"original_annotated_actions":sum(action_rows.values()),
                     "original_unseen_source_pairs_by_domain":dict(distribution),
                     "blocked_original_source_task_shas":len(blocked_tasks),
                     "source_task_sha_excluded":True,
                     "min_source_coord_target_ratio":min(ratios),
                     "max_source_coord_target_ratio":max(ratios),
                     "excluded_candidate_reasons":dict(dropped)}
    finally:
        model.only_input_file=old
        for z in extras:z.close()
        if zips:
            for value in zips.values():
                if isinstance(value,zipfile.ZipFile):value.close()


def export(rows,meta,path):
    p=Path(path);p.mkdir(parents=True,exist_ok=True)
    target=p/OUT;receipt=p/RECEIPT;side=p/"receipt.sha256"
    if target.exists() or receipt.exists() or side.exists():
        raise FileExistsError("H421 private source test heldout export create only")
    content=b"".join(canonical(row)+b"\n" for row in rows)
    rec={
      "schema":"hala.h421.source-original-counterfactual-dev.v1",
      "status":"SOURCE_ORIGINAL_192_PAIR_384_NATIVE_ACTION_DEV_NOT_MODEL_EVALUATED",
      "source":"Hcompany/DragOn original H60 TRAIN not benchmark eval",
      "source_revision":model.SOURCE_REV,
      "owner_heldout_source_image_pairs":TARGET_PAIRS,
      "native_two_instruction_eval_cases":REQUIRED_ACTIONS,
      "native_pairs_by_source_family":meta["original_unseen_source_pairs_by_domain"],
      "frozen_original_dev_rows_sha256":sha256(content).hexdigest(),
      "nine_original_source_zip_sha256":model.H419_ALL_SOURCE_SHAS,
      "excluded_h417_and_h419_train_and_dev_source_images":True,
      "excluded_source_task_sha_of_h417_h419_models_and_dev":meta["source_task_sha_excluded"],
      "original_jpeg_media_sha_replayed":True,
      "source_target_different_instructions_and_coordinates":True,
      "source_original_action_counts":meta["original_annotated_actions"],
      "min_source_coord_target_ratio":meta["min_source_coord_target_ratio"],
      "independent_external_ood":False,
      "model_evaluation_executed":False,
      "model_training_executed":False,
      "model_release_qualified":False,
    }
    try:
        with target.open("xb") as f:f.write(content)
        raw=json.dumps(rec,sort_keys=True,indent=2,allow_nan=False).encode()
        with receipt.open("xb") as f:f.write(raw)
        with side.open("x",encoding="ascii",newline="\n") as f:
            f.write(sha256(raw).hexdigest())
    except BaseException:
        target.unlink(missing_ok=True);receipt.unlink(missing_ok=True);side.unlink(missing_ok=True)
        raise
    return rec


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    rows,meta=audit(a.source_root)
    print("H421_ORIGINAL_SOURCE_DEV",
          json.dumps(export(rows,meta,a.out),sort_keys=True),flush=True)


if __name__=="__main__":
    run()
