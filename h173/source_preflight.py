"""H173: freeze two truly unseen MIT original GUI OOD application source IDs.

This is a SOURCE-only gate. It NEVER opens historical OOD pixels, trains,
scores, produces model inferences, or approves unreplayed source screenshots.
"""
import hashlib
import subprocess
from pathlib import Path

SCHEMA="hala.h173.two-genuinely-unseen-original-app-ood-source.v1"
SOURCE_PINS={
    "tingle": {
        "git_sha":"9eadca70c72ef9adb9144cc23a4104978d3bc63b",
        "original_html":"doc/index.html",
        "original_html_sha256":"19ff3e667949ec52196b404c766c06f061004da5949057b589bc7bcfe4d388ea",
        "license":"LICENSE",
        "action":"click",
        "family":"native-modal-visibility",
        "scenario":"open-original-modal"
    },
    "tiny-slider": {
        "git_sha":"4d709735c417c2483e77a22d017fc1b18c04f0d4",
        "original_html":"demo/index.html",
        "original_html_sha256":"76f433876dc2fd5373c62460c9e2151701d8b3c320e386c10337805a03bc8e2e",
        "license":"LICENSE",
        "action":"click",
        "family":"native-carousel-index",
        "scenario":"advance-original-carousel"
    }
}
TRAIN_AND_CONSUMED_FAMILIES=frozenset({
    "choices","nouislider","flatpickr","awesomplete",
    "pell","pikaday","todomvc","dnd-kit","simpletable",
    "calculator","sortablejs"
})
OOD_COUNT=32
VIEWPORT=(960,720)

def sha(blob):return hashlib.sha256(blob).hexdigest()

def verified_original_source(upstream):
    upstream=Path(upstream)
    if len(set(SOURCE_PINS)&TRAIN_AND_CONSUMED_FAMILIES)!=0:
        raise ValueError("H173 new OOD source is already a known TRAIN or consumed OOD app")
    qualified={}
    for name,pin in SOURCE_PINS.items():
        root=upstream/name
        if not root.is_dir():
            raise ValueError("H173 original upstream missing: "+name)
        git_sha=subprocess.check_output(
            ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(
            ["git","-C",str(root),"status","--porcelain"],text=True).strip()
        if git_sha!=pin["git_sha"] or dirty:
            raise ValueError("H173 upstream Git SHA/dirty source rejected: "+name)
        license_text=(root/pin["license"]).read_text(encoding="utf8")
        if "MIT" not in license_text[:60]:
            raise ValueError("H173 original upstream MIT license not present: "+name)
        html=(root/pin["original_html"]).read_bytes()
        if sha(html)!=pin["original_html_sha256"]:
            raise ValueError("H173 exact original HTML SHA changed: "+name)
        qualified[name]=dict(
            original_git_sha=git_sha,
            original_html_sha256=sha(html),
            source_license="MIT",
            action_type=pin["action"],native_state_family=pin["family"],
            source_role="OOD_CANDIDATE_NOT_EVALUATED",
            model_gpu_calls=0,reference_labels_uploaded=False)
    if len({v["native_state_family"] for v in qualified.values()})!=2:
        raise ValueError("H173 two independent interaction families required")
    return qualified

def preregister_provisional_cases():
    """Eligibility roster only: must still pass native QA and pixel replay."""
    specs=[]
    for app in SOURCE_PINS:
        for n in range(16):
            specs.append(dict(
                index=173000+(0 if app=="tingle" else 100)+n,
                split="OOD_CANDIDATE_NOT_FROZEN",
                app=app,
                family=SOURCE_PINS[app]["family"],
                task=SOURCE_PINS[app]["scenario"],
                action_type="click",
                variant=n,
                language="en" if n%2==0 else "ar"))
    if (len(specs)!=OOD_COUNT or len({s["index"] for s in specs})!=OOD_COUNT
        or sum(s["language"]=="ar" for s in specs)!=16):
        raise ValueError("H173 OOD provisional source schedule malformed")
    return specs

def assert_ood_not_qualified_without_new_receipts(metadata):
    """Fail closed: a clone/smoke does NOT equal scored model OOD."""
    if not isinstance(metadata,dict):
        raise ValueError("H173 new original OOD status missing")
    if (metadata.get("new_ood_gold_free_source_frozen") is not True or
        metadata.get("independent_pixel_replays")!=OOD_COUNT or
        metadata.get("native_reference_successes")!=OOD_COUNT or
        metadata.get("before_model_inferences")!=OOD_COUNT or
        metadata.get("after_model_inferences")!=OOD_COUNT or
        metadata.get("native_one_shot_rewards")!=OOD_COUNT*2 or
        metadata.get("old_eval_used") is not False or
        metadata.get("dev_gold_on_gpu") is not False or
        metadata.get("paid_gpu") is not False):
        raise ValueError("H173 source smoke alone cannot satisfy unseen-app OOD quality gate")
    return True
