"""Independent ORIGINAL 384-case H421 model evaluation verifier for true H425.

Runs CPU ONLY. Recalculates real native DRAG instruction sensitivity against
SHA-frozen original source labels. Does not use H419 rejected final or claim
sealed external bilingual OOD. All exact predictions stay owner PRIVATE.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location("h425_h421_frozen_original_eval",
    ROOT/"h425/kaggle/h425_h421_paired_eval.py")
h=importlib.util.module_from_spec(s)
s.loader.exec_module(h)
SCHEMA="hala.h425.original-paired-h421-384-model-eval.v1"
COMPLETE="COMPLETE_PRIVATE_H417_VS_H425_TRUE_ORIGINAL_PAIRED"
FAILED="FAILED_PRIVATE_H425_H421_MODEL_EVAL_NO_RELEASE"

def _sha(value:bytes):
    return hashlib.sha256(value).hexdigest()

def _digest(value):
    return type(value) is str and len(value)==64 and all(ch in "0123456789abcdef" for ch in value)

def _validate_metadata(data):
    if (type(data) is not dict
        or data.get("schema")!=SCHEMA
        or data.get("status")!=COMPLETE
        or data.get("official_base")!=h.BASE
        or data.get("official_revision")!=h.REV
        or data.get("h417_parent_adapter_sha256")!=h.PARENT_ADAPTER_SHA
        or data.get("source_pairs_jsonl_sha256")!=h.SOURCE_JSONL_SHA
        or data.get("source_pairs_receipt_sha256")!=h.SOURCE_RECEIPT_SHA
        or data.get("h425_completed_source_optimizer_steps")!=12696
        or data.get("h425_source_only_research_gate_passed") is not True
        or data.get("original_heldout_images")!=192
        or data.get("native_contrastive_original_instructions")!=384
        or data.get("device_name")!="Tesla T4"
        or not _digest(data.get("h425_authentic_trained_adapter_sha256"))
        or not _digest(data.get("h425_authentic_private_receipt_sha256"))
        or data.get("model_training_executed_in_this_kernel") is not False
        or data.get("independent_external_ood") is not False
        or data.get("model_release_qualified") is not False
        or data.get("weights_public") is not False
        or type(data.get("research_pair_improved")) is not bool):
        raise ValueError("H425 H421 original model pair evidence/rights/release metadata invalid")

def verify(receipt_root,labels,source_receipt,source_sidecar,
           adapter_file=None,h425_model_receipt=None):
    root=Path(receipt_root)
    raw=(root/"receipt.json").read_bytes()
    digest=_sha(raw)
    if (root/"receipt.sha256").read_text("ascii").strip()!=digest:
        raise ValueError("H425 H421 real counterfactual GPU receipt SHA sidecar changed")
    data=json.loads(raw)
    if data.get("status")==FAILED:
        if type(data.get("failure_class")) is not str:
            raise ValueError("H425 real counterfactual model failure lacks error class")
        return {"status":FAILED,"failure_class":data["failure_class"],
                "receipt_sha256":digest,"model_release_qualified":False}
    _validate_metadata(data)
    frozen=h.load_sources(labels,source_receipt,source_sidecar)
    rows=[case for pair in range(h.IMAGE_PAIRS)
          for case in (frozen[pair][0],frozen[pair][1])]
    observed={}
    for key in ("h417_parent","h425_trained"):
        trace=data.get(key+"_exact_predictions")
        if type(trace) is not list or len(trace)!=384:
            raise ValueError("H425 original 384 genuine model-generated predictions incomplete")
        stats=h.calculate(rows,trace)
        if data.get(key+"_pair_stats")!=stats:
            raise ValueError("H425 H421 native two-instruction original source scoring forged")
        observed[key]=stats
    true_gate=observed["h425_trained"]["both_actions_correct"]>(
              observed["h417_parent"]["both_actions_correct"])
    if data["research_pair_improved"] is not true_gate:
        raise ValueError("H425 original counterfactual model pair improvement gate forged")
    if adapter_file is not None:
        actual=Path(adapter_file).read_bytes()
        if _sha(actual)!=data["h425_authentic_trained_adapter_sha256"]:
            raise ValueError("H425 original trained model private LoRA adapter SHA changed")
    child_verified=False
    if h425_model_receipt is not None:
        child_file=Path(h425_model_receipt)
        original=child_file.read_bytes()
        if _sha(original)!=data["h425_authentic_private_receipt_sha256"]:
            raise ValueError("H425 true source-trained model private receipt SHA mismatch")
        from importlib.util import spec_from_file_location,module_from_spec
        sp=spec_from_file_location("h425_trained_receipt_validator",
                                   ROOT/"h425/validate_receipt.py")
        validator=module_from_spec(sp)
        sp.loader.exec_module(validator)
        child=json.loads(original)
        _oldp,_freshp,_old,_fresh,gate,strong=validator._eval_contract(child)
        if not gate or not strong or child["model_weights_sha256"]!=(
                data["h425_authentic_trained_adapter_sha256"]):
            raise ValueError("H425 genuinely trained model child source-only gate/weights not qualified")
        child_verified=True
    return {
        "status":COMPLETE,"receipt_sha256":digest,
        "original_heldout_images":192,"source_original_instruction_cases":384,
        "h417_parent":observed["h417_parent"],
        "h425_trained":observed["h425_trained"],
        "source_only_pair_improved":true_gate,
        "adapter_sha_verified":adapter_file is not None,
        "source_trained_child_receipt_validated":child_verified,
        "genuinely_independent_external_ood":False,
        "model_release_qualified":False}

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--receipt-root",type=Path,required=True)
    cli.add_argument("--labels",type=Path,required=True)
    cli.add_argument("--source-receipt",type=Path,required=True)
    cli.add_argument("--source-sidecar",type=Path,required=True)
    cli.add_argument("--adapter",type=Path)
    cli.add_argument("--h425-model-receipt",type=Path)
    args=cli.parse_args()
    print(json.dumps(verify(
        args.receipt_root,args.labels,args.source_receipt,args.source_sidecar,
        args.adapter,args.h425_model_receipt),sort_keys=True))
