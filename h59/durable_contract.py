"""MiniMax/Vercel H59 durable state admission: pure, CPU-only, zero provider side effects.

Caller must independently check actual provider receipts. A transition's
attestation cannot, by itself, prove that Kaggle/GPU work physically happened.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from h59.master_data_plan import REPO,REV,check_plan

SCHEMA="hala.h59.durable-window-job-state.v1"
STATES=("PLANNED","WINDOW_SHA_VERIFIED","ORIGINAL_MEMBERS_VERIFIED",
        "PRIVATE_KAGGLE_READBACK_VERIFIED")
SHA=re.compile(r"^[0-9a-f]{64}$")
OWNER="abdulazizshehri"

def key(entry):
    if not isinstance(entry,dict) or not isinstance(entry.get("id"),str):
        raise ValueError("Window identity missing")
    identity=f"{REPO}|{REV}|{entry['source_domain']}|{entry['byte_start']}|{entry['byte_end_inclusive']}|h59.v1"
    return hashlib.sha256(identity.encode("utf8")).hexdigest()

def initial(entry):
    return dict(schema=SCHEMA,job_id=key(entry),window_id=entry["id"],
        source_repo=REPO,source_revision=REV,
        source_byte_start=entry["byte_start"],source_byte_end_inclusive=entry["byte_end_inclusive"],
        state="PLANNED",original_window_sha256=None,original_member_manifest_sha256=None,
        original_member_count=0,original_source_pair_count=0,
        private_kaggle_ref=None,private_provider_readback_sha256=None,
        private_provider_is_private_verified=False,anonymous_provider_http403_verified=False,
        authorized_free_compute=False,model_gpu_steps=0,
        model_training_qualified=False,independent_neural_evaluation_completed=False)

def verify_identity(previous,updated,entry):
    basic=initial(entry)
    pinned_fields=("schema","job_id","window_id","source_repo","source_revision",
                   "source_byte_start","source_byte_end_inclusive")
    for field in pinned_fields:
        if previous.get(field)!=basic[field] or updated.get(field)!=basic[field]:
            raise ValueError("Source identity has changed across durable retries")
    if previous.get("state") not in STATES or updated.get("state") not in STATES:
        raise ValueError("Unrecognized lifecycle status")
    if (updated.get("model_gpu_steps")!=0 or
        updated.get("model_training_qualified") is not False or
        updated.get("independent_neural_evaluation_completed") is not False or
        updated.get("authorized_free_compute") is not False):
        raise ValueError("Dataset workflow may never masquerade as a GPU training job")
    if STATES.index(updated["state"])<STATES.index(previous["state"]):
        raise ValueError("Original source workflow cannot regress completed states")
    if STATES.index(updated["state"])>STATES.index(previous["state"])+1:
        raise ValueError("No skipping intermediate evidence gates")
    for field in ("original_window_sha256","original_member_manifest_sha256",
                  "private_kaggle_ref","private_provider_readback_sha256"):
        if previous.get(field) is not None and updated.get(field)!=previous[field]:
            raise ValueError("Immutable original proof cannot be rewritten")
    for field in ("original_member_count","original_source_pair_count"):
        if (type(updated.get(field)) is not int or updated[field]<0 or
            updated[field]<previous.get(field,0)):
            raise ValueError("Negative/regressing source counts not permitted")
    return basic

def validate_transition(previous,updated,entry):
    verify_identity(previous,updated,entry)
    stage=STATES.index(updated["state"])
    if stage>=1:
        if (not isinstance(updated.get("original_window_sha256"),str) or
            not SHA.fullmatch(updated["original_window_sha256"])):
            raise ValueError("Only actual original verified HTTP window SHA receipt may advance")
    if stage>=2:
        if (not isinstance(updated.get("original_member_manifest_sha256"),str) or
            not SHA.fullmatch(updated["original_member_manifest_sha256"]) or
            updated["original_member_count"]<1 or
            updated["original_source_pair_count"]<1):
            raise ValueError("Original TAR member reconstruction and real source pair evidence missing")
    if stage>=3:
        ref=updated.get("private_kaggle_ref")
        if (not isinstance(ref,str) or not ref.startswith(OWNER+"/") or
            ref.lower().endswith("-public") or
            not isinstance(updated.get("private_provider_readback_sha256"),str) or
            not SHA.fullmatch(updated["private_provider_readback_sha256"]) or
            updated["private_provider_readback_sha256"]!=updated["original_member_manifest_sha256"] or
            updated.get("private_provider_is_private_verified") is not True or
            updated.get("anonymous_provider_http403_verified") is not True):
            raise ValueError("Private provider SHA, authentication/403 and namespace not qualified")
        # This pure validator cannot inspect provider is_private or HTTP403; caller
        # must collect independent live provider response and attach receipt.
    return dict(job_id=updated["job_id"],status=updated["state"],
      provider_private_status_not_independently_proven_here=True,
      model_training_in_progress=False,training_complete=False)

def check_manifest_jobs(plan,records):
    check_plan(plan)
    if not isinstance(records,list) or len(records)>plan["planned_windows"]:
        raise ValueError("No duplicate/orphan source jobs")
    windows={p["id"]:p for p in plan["windows"]}
    keys=set()
    for record in records:
        if not isinstance(record,dict):
            raise ValueError("Bad original source job receipt")
        wid=record.get("window_id")
        if wid not in windows or record.get("job_id") in keys:
            raise ValueError("Duplicate or unplanned original window state")
        keys.add(record["job_id"])
        previous=initial(windows[wid])
        for state in STATES[:STATES.index(record["state"])+1]:
            candidate={**record,"state":state}
            validate_transition(previous,candidate,windows[wid])
            previous=candidate
    return dict(planned_windows=len(windows),registered_jobs=len(records),
                unregistered_windows=len(windows)-len(records),
                real_training_steps=0)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--plan",default="h59/full_source_windows.json")
    p.add_argument("--jobs")
    args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text("utf8"))
    records=[] if args.jobs is None else json.loads(Path(args.jobs).read_text("utf8"))
    print(json.dumps(check_manifest_jobs(plan,records),sort_keys=True))
