"""H421 ORIGINAL private GPU pair-eval receipt verifier.

Recompute 384 paired native-DRAG predictions for actual H417 parent and
actual fully trained H419 LoRA using the SHA-frozen 192 unseen original
screens / two source-original prompts per image.

No third-party independent OOD, release signoff or production actions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from h420 import counterfactual_score as score
from h419.kaggle import h419_native_drag_continuation as parent

RESEARCH_STATUS="COMPLETE_PRIVATE_H417_VS_H419_192_ORIGINAL_PAIR_EVAL"
FAIL="FAILED_PRIVATE_COUNTERFACTUAL_MODEL_EVAL_NO_RELEASE"
SCHEMA="hala.h421.private-two-instruction-model-paired-eval.v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path,labels,source_receipt,source_sidecar,actual_h419_weights=None):
    path=Path(path)
    raw=(path/"receipt.json").read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if (path/"receipt.sha256").read_text("ascii").strip()!=digest:
        raise ValueError("H421 original GPU receipt SHA sidecar mismatch")
    data=json.loads(raw)
    if data.get("status")==FAIL:
        if not isinstance(data.get("failure_class"),str):
            raise ValueError("H421 failure receipt missing original error class")
        return {"status":FAIL,"receipt_sha256":digest,"qualified":False,
                "failure_class":data["failure_class"]}
    if (data.get("schema")!=SCHEMA or data.get("status")!=RESEARCH_STATUS
        or data.get("official_base")!=parent.BASE
        or data.get("official_revision")!=parent.REV
        or data.get("h417_parent_adapter_sha256")!=parent.PARENT_ADAPTER_SHA
        or data.get("source_pairs_jsonl_sha256")!=score.SOURCE_JSONL_SHA
        or data.get("source_pairs_receipt_sha256")!=score.SOURCE_RECEIPT_SHA
        or data.get("h419_real_optimizer_steps")!=parent.H419_STEPS
        or data.get("h419_real_gradient_examples")!=parent.H419_TRAINING_PASSES
        or data.get("paired_original_images")!=score.IMAGE_PAIRS
        or data.get("original_source_actions")!=score.SAMPLE_N
        or data.get("device_name")!="Tesla T4"
        or data.get("model_training_executed") is not False
        or data.get("independent_external_ood") is not False
        or data.get("model_release_qualified") is not False
        or data.get("weights_public") is not False
        or type(data.get("research_pair_improved")) is not bool
        or type(data.get("h419_train_source_only_research_gate")) is not bool):
        raise ValueError("H421 cannot claim genuine source-only two-model GPU research")
    child_sha=data.get("h419_trained_adapter_sha256")
    child_receipt=data.get("h419_trained_private_receipt_sha256")
    if (type(child_sha) is not str or len(child_sha)!=64 or
        type(child_receipt) is not str or len(child_receipt)!=64 or
        any(ch not in "0123456789abcdef" for ch in child_sha+child_receipt)):
        raise ValueError("H421 actual H419 trained LoRA and source receipt SHA invalid")
    if actual_h419_weights is not None:
        if sha(actual_h419_weights)!=child_sha:
            raise ValueError("H421 wrong actual H419 trained adapter bytes")
    pairs=score.load_sources(labels,source_receipt,source_sidecar)
    ordered=[x for i in range(score.IMAGE_PAIRS) for x in
             (pairs[i][0],pairs[i][1])]
    actual={}
    for name in ("h417_parent","h419_trained"):
        predictions=data.get(name+"_exact_predictions")
        expected_metrics=data.get(name+"_pair_stats")
        if type(predictions) is not list or len(predictions)!=score.SAMPLE_N:
            raise ValueError("H421 model real 384 native predictions missing")
        computed=score.calculate(ordered,predictions)
        if computed!=expected_metrics:
            raise ValueError("H421 model paired correctness or collapse metrics forged")
        actual[name]=computed
    if data["research_pair_improved"] is not (
            actual["h419_trained"]["both_actions_correct"]>
            actual["h417_parent"]["both_actions_correct"]):
        raise ValueError("H421 model source-only paired research gate not reproducible")
    return {"status":RESEARCH_STATUS,
            "receipt_sha256":digest,
            "h419_actual_adapter_sha256":child_sha,
            "h419_adapter_bytes_sha_verified":actual_h419_weights is not None,
            "original_192_pairs_replayed":True,
            "h417_parent":actual["h417_parent"],
            "h419_trained":actual["h419_trained"],
            "research_pair_improved":data["research_pair_improved"],
            "independent_external_ood":False,
            "qualified":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--receipt-root",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--source-receipt",required=True)
    p.add_argument("--source-sidecar",required=True)
    p.add_argument("--adapter",default=None)
    a=p.parse_args()
    print(json.dumps(verify(a.receipt_root,a.labels,a.source_receipt,
                            a.source_sidecar,a.adapter),sort_keys=True))
