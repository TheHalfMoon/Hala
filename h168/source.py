"""H168 private new upstream source qualification and sealed TRAIN-pilot protocol.

No H164 or historical test screens, labels, or OOD calls. Original app demos
are served directly from their pinned upstream tree, not reimplemented HTML.
"""
import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA="hala.h168.new-original-app-train-pilot.v1"
PINS={
    "choices":("e132628ac74dd427a9a454caf042c65d3065d170","LICENSE",
               "public/test/text/index.html","type"),
    "nouislider":("57033672d07e8bac527d7147721a4a2e07e0d699","LICENSE.md",
                  "tests/origin.html","drag"),
    "flatpickr":("548330e6e18871d6ca1ca78746c0e89cb12a0303","LICENSE.md",
                 "index.template.html","click"),
}
APPS_FOR_PILOT=("choices","nouislider")
N_PER_APP=16
W=960
H=720

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def verify_sources(upstream):
    upstream=Path(upstream)
    proof={}
    for name,(pin,license_file,html,action) in PINS.items():
        root=upstream/name
        if not root.is_dir():
            raise ValueError("H168 original upstream source is missing: "+name)
        commit=subprocess.check_output(
            ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        clean=subprocess.check_output(
            ["git","-C",str(root),"status","--porcelain"],text=True).strip()
        license_text=(root/license_file).read_text("utf8")
        if commit!=pin or clean or not license_text.lstrip().startswith(
                ("MIT License","The MIT License")):
            raise ValueError("H168 upstream commit cleanliness or MIT license failed: "+name)
        html_file=root/html
        if not html_file.is_file():
            raise ValueError("H168 original unmodified app demo missing")
        proof[name]=dict(app=name,git_sha=commit,license="MIT",
                         license_file=license_file,
                         license_sha256=sha((root/license_file).read_bytes()),
                         original_html_path=html,
                         original_html_sha256=sha(html_file.read_bytes()),
                         native_action=action,
                         pilot_role="TRAIN_ONLY" if name in APPS_FOR_PILOT
                                   else "SOURCE_CANDIDATE_NOT_TRAINED")
    return proof

def historical_image_shas(owner):
    """Only historical SHA manifests, never read/inspect held-out screenshots."""
    from h164.protocol import locked_prior_shas
    owner=Path(owner)
    existing=set(locked_prior_shas(owner))
    for manifest in sorted((owner/"private-h164").rglob("manifest.json")):
        rows=json.loads(manifest.read_text("utf8")).get("private_cases",[])
        if not isinstance(rows,list):
            raise ValueError("H168 historic source manifest malformed")
        for row in rows:
            digest=row.get("image_sha256")
            if (not isinstance(digest,str) or len(digest)!=64 or
                any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError("H168 invalid prior original-app image digest")
            existing.add(digest)
    if len(existing)<120:
        raise ValueError("H168 previous used app SHA registry unexpectedly sparse")
    return existing

def cases():
    out=[]
    for app in APPS_FOR_PILOT:
        action=PINS[app][3]
        for n in range(N_PER_APP):
            out.append(dict(split="TRAIN_PILOT",app=app,
                            task="add_new_tag" if app=="choices" else
                                 "drag_slider_handle",
                            action_type=action,language="en" if n%2==0 else "ar",
                            index=168000+(0 if app=="choices" else 100)+n,
                            variant=n))
    if len(out)!=32 or len({c["index"] for c in out})!=32:
        raise ValueError("H168 source schedule not qualified")
    return out

def validate_pilot(manifest,supervision,prior):
    """Strict input/gold separation, no reused SHA, proven native state only."""
    if (manifest.get("schema")!=SCHEMA or manifest.get("split")!="TRAIN_PILOT"
        or manifest.get("count")!=len(manifest.get("private_cases",[]))):
        raise ValueError("H168 TRAIN pilot manifest identity invalid")
    scenes=manifest["private_cases"]
    records=supervision.get("private_training_actions")
    if not isinstance(records,list) or len(records)!=len(scenes):
        raise ValueError("H168 private reference action count mismatch")
    ids=set()
    seen=set(prior)
    for scene,gold in zip(scenes,records):
        digest=scene.get("image_sha256")
        if (not isinstance(digest,str) or len(digest)!=64 or
            any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("H168 invalid exact SHA-256 source identity")
        if (scene.get("index") in ids or scene.get("image_sha256") in seen
            or scene.get("action_type") not in ("type","drag")
            or scene.get("language") not in ("en","ar")
            or scene.get("split")!="TRAIN_PILOT"
            or scene.get("app") not in APPS_FOR_PILOT
            or not scene.get("native_browser_state_verified")
            or any(k in scene for k in ("training_supervision","gold","reference_point"))
            or gold.get("index")!=scene.get("index")
            or gold.get("kind")!=scene.get("action_type")
            or gold.get("native_success") is not True):
            raise ValueError("H168 private split integrity, action, or source-native reward invalid")
        ids.add(scene["index"])
        seen.add(scene["image_sha256"])
        if scene["action_type"]=="type":
            if (not isinstance(gold.get("text"),str) or
                not isinstance(gold.get("point"),list) or len(gold["point"])!=2):
                raise ValueError("H168 native type supervision invalid")
        else:
            if (not isinstance(gold.get("start"),list) or len(gold["start"])!=2
                or not isinstance(gold.get("end"),list) or len(gold["end"])!=2):
                raise ValueError("H168 native drag supervision invalid")
    return dict(status="H168_FRESH_ORIGINAL_TRAIN_PILOT_SOURCE_ADMITTED",
                distinct_apps=len({r["app"] for r in scenes}),
                cases=len(scenes),input_gold_separated=True,
                prior_consumed_source_excluded=True,
                dev_ood_evaluations=0,optimizer_steps=0,
                paid_gpu=False)
