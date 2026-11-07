"""H426: audit genuine H425/H417 H421 errors, aggregates ONLY, no source/predictions output."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("h426_replay", ROOT/"h425/validate_h421_pair_receipt.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
h=v.h

def breakdown(rows, traces):
    by_id={x["id"]:x for x in traces}
    if len(traces)!=384 or len(by_id)!=384:raise ValueError("384 unique original predictions required")
    by=collections.defaultdict(collections.Counter)
    hits=[]
    for case in rows:
        p=by_id.get(case["id"])
        if p is None or p["image_sha256"]!=case["image_sha256"]:
            raise ValueError("H421 original source image/prompt drift")
        gold=case["native_drag_target"]
        scored=h.score_predicted(p["raw_prediction"],gold)
        pred=h.parse_drag(p["raw_prediction"])
        c=by[case["source_domain"]]
        c["total"]+=1
        c["valid"]+=int(scored["valid"])
        c["correct"]+=int(scored["correct"])
        hits.append(bool(scored["correct"]))
        if pred is None:c["schema_failure"]+=1;continue
        if pred["ordered"]!=gold["ordered"]:c["order_failure"]+=1;continue
        start=max(abs(a-b) for a,b in zip(pred["start_point"],gold["start_point"]))
        end=max(abs(a-b) for a,b in zip(pred["end_point"],gold["end_point"]))
        a=start<=h.TOL_NORM
        b=end<=h.TOL_NORM
        c["start_near"]+=int(a)
        c["end_near"]+=int(b)
        if scored["correct"]:continue
        if a and not b:c["end_point_only_failure"]+=1
        elif b and not a:c["start_point_only_failure"]+=1
        elif not a and not b:c["both_point_failure"]+=1
        else:c["swap_or_order_scoring_failure"]+=1
        c["failed_start_within_2xtol"]+=int(start<=2*h.TOL_NORM)
        c["failed_end_within_2xtol"]+=int(end<=2*h.TOL_NORM)
    return {k:dict(v) for k,v in sorted(by.items())}, hits

def audit(*, pair_root, labels, source_receipt, source_sidecar, adapter, child_receipt):
    proof=v.verify(pair_root,labels,source_receipt,source_sidecar,adapter,child_receipt)
    if not (proof["source_only_pair_improved"] and proof["adapter_sha_verified"]
            and proof["source_trained_child_receipt_validated"]
            and proof["genuinely_independent_external_ood"] is False):
        raise ValueError("Authentic H425 model and original H421 pair proof required")
    source=h.load_sources(labels,source_receipt,source_sidecar)
    cases=[case for i in range(192) for case in (source[i][0],source[i][1])]
    actual=json.loads((pair_root/"receipt.json").read_bytes())
    parent,p_hits=breakdown(cases,actual["h417_parent_exact_predictions"])
    trained,t_hits=breakdown(cases,actual["h425_trained_exact_predictions"])
    changes=collections.Counter()
    for pair in range(192):
        i=2*pair
        before=int(p_hits[i])+int(p_hits[i+1])
        after=int(t_hits[i])+int(t_hits[i+1])
        changes[f"{before}_to_{after}"]+=1
    if (sum(v["correct"] for v in parent.values())!=86 or
        sum(v["correct"] for v in trained.values())!=172 or
        len(p_hits)!=384 or len(t_hits)!=384 or sum(changes.values())!=192):
        raise ValueError("Real provider H421 aggregate mismatch")
    for label,agg in (("h417_parent",parent),("h425_trained",trained)):
        for dom,stats in agg.items():
            if stats["correct"]!=proof[label]["by_domain"][dom]["single_correct"]:
                raise ValueError("H421 original source evaluator family score drift")
    return dict(schema="hala.h426.private-h421-diagnostic-aggregates.v1",
                evaluation="source_author_heldout_not_external",
                independent_external_ood=False,model_release_qualified=False,
                individual_predictions_included=False,
                h421_receipt_sha256=hashlib.sha256((pair_root/"receipt.json").read_bytes()).hexdigest(),
                source_images=192,source_actions=384,
                tolerance_normalized_1000=h.TOL_NORM,
                h417_parent=parent,h425_trained=trained,
                pairs_correct_transition=dict(sorted(changes.items())))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for a in ("pair-root","labels","source-receipt","source-sidecar","adapter","child-receipt"):
        p.add_argument("--"+a,required=True,type=Path)
    print(json.dumps(audit(**{k.replace("-","_"):v for k,v in vars(p.parse_args()).items()}),
                     sort_keys=True,indent=2))
