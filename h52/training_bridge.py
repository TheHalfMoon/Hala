"""Training-only H52 adapter bridge with pinned original browser source.

No GPU or inference runs occur here. The returned training rows are genuine
source-bound first-party screenshots/targets, never independent model results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from h52.audit import audit
from h52.factory import VERSION, FAMILIES
from training.artifacts import strict_json

SOURCE_SHA256="564ecb32075859fa12a15c2fe488bdc68cc32e0207d878307244c891dab42d75"
DATASET_SHA256="bca090f0a31aee88fba0cf47aaadaa895ca1fc7ede7c41fbf123d65d6ccb4c73"
FAMILIES_TRAIN=list(FAMILIES)
# These are EXISTING public DEV families in H49, not qualified blind 3-step TEST.
PUBLIC_DEV_FAMILIES=["municipal_portal","research_lab"]
ADDITIONAL_FIELDS={"families":FAMILIES_TRAIN,"held_out_families":PUBLIC_DEV_FAMILIES}


def expected_training_manifest(root):
    root=Path(root)
    report=audit(root)
    original=strict_json((root/"sft_manifest.json").read_text(encoding="utf-8"))
    source=strict_json((root/"source.json").read_text(encoding="utf-8"))
    if (original.get("source_schema")!=VERSION or
        original.get("source_sha256")!=SOURCE_SHA256 or
        original.get("dataset_sha256")!=DATASET_SHA256 or
        source.get("benchmark_eval_included") is not False or
        source.get("external_dataset_media") is not False or
        original.get("model_training_executed") is not False or
        report["rows"]!=288 or report["original_workflows"]!=96 or
        report["language_rows"]!={"ar":144,"en":144}):
        raise ValueError("Unqualified or altered H52 original training source")
    if set(original)&set(ADDITIONAL_FIELDS):
        raise ValueError("H52 original manifest cannot already declare a derived training split")
    return {**original,**ADDITIONAL_FIELDS}


def load_export(examples_path,manifest):
    examples_path=Path(examples_path)
    if (examples_path.name!="authoring_sft.jsonl" or
        examples_path.is_symlink() or
        not isinstance(manifest,dict)):
        raise ValueError("Only original H52 source-verified train annotations admitted")
    root=examples_path.parent
    expected=expected_training_manifest(root)
    if manifest!=expected:
        raise ValueError("H52 remote training manifest does not match frozen original evidence")
    # Full byte-level audit rechecks every screenshot/action and original Edge receipt.
    # Never interpret the mock-browser clicks as neural end-to-end outcomes.
    return [strict_json(line) for line in examples_path.read_text("utf-8").splitlines()]


def emit_manifest(root,output):
    dest=Path(output)
    if dest.exists():
        raise FileExistsError("Only create a new versioned training manifest")
    expected=expected_training_manifest(root)
    if dest.parent and not dest.parent.is_dir():
        raise FileNotFoundError("Parent must exist: "+str(dest.parent))
    with dest.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(expected,f,indent=2,ensure_ascii=False,sort_keys=True)
        f.write("\n")
    return dict(train_rows=288,real_model_training_executed=False,
                source_sha256=SOURCE_SHA256,training_manifest=str(dest),
                public_dev_only=PUBLIC_DEV_FAMILIES)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="h52/assets")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(emit_manifest(args.root,args.output),sort_keys=True))
