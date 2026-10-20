"""H38 offline, SHA-bound post-hoc *filter* of original H37 model outputs.

This does not demonstrate new neural performance, external-app accuracy, or
native action permission. The H37 oracle remains evaluator-only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from h34.geometry import bind_point
from h34.protocol import parse_raw, score_attempt
from h37.audit import audit as audit_h37
from hala.inference.explicit_label_guard import unique_exact_label_match

RAW_SHA = "40b88ceb592446630817b9562238b40765b371fac61ccb4c06b427fab92cef1d"
AUDIT_SHA = "987c8338342faaf8d2718288623a9f485d058acdde0f21f5f3133befaa1451f2"
SOURCE_SHA = "4c061e9c2e15df0306062b5a5666c6e0d7697502"


def analyze(evidence: Path) -> dict:
    evidence = Path(evidence)
    journal = (evidence / "raw_attempts.jsonl").read_bytes()
    audit_bytes = (evidence / "audit.json").read_bytes()
    if hashlib.sha256(journal).hexdigest() != RAW_SHA:
        raise ValueError("original H37 raw SHA differs")
    if hashlib.sha256(audit_bytes).hexdigest() != AUDIT_SHA:
        raise ValueError("original H37 audit SHA differs")
    prior = audit_h37(evidence, RAW_SHA)
    if (not prior["complete"] or prior["attempts"] != 64 or
            prior["source_commit_observed"] != SOURCE_SHA or
            prior["totals"]["target_hit"] != 32 or
            prior["totals"]["wrong_target"] != 16):
        raise ValueError("H37 source or original measurements changed")
    recomputed = (json.dumps(prior, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode("utf-8")
    if recomputed != audit_bytes:
        raise ValueError("recalculated original H37 audit differs")
    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / "h37/assets/evaluator_only.json").read_text("utf-8"))
    records = [json.loads(line) for line in journal.decode("utf-8").splitlines()]
    if len(cases) != 64 or len(records) != 64:
        raise ValueError("incomplete H37 cases or raw responses")
    counts = Counter()
    langs = {v:Counter() for v in ("en","ar")}
    layouts = {v:Counter() for v in ("toolbar","sidebar","grid","table")}
    result_rows = []
    for case, record in zip(cases,records):
        if (record["case_id"] != case["id"] or
                record["image_sha256"] != case["image_sha256"]):
            raise ValueError("model attempt bound to different screenshot")
        prior_score = score_attempt(case,record["raw"],record["error"])
        point = parse_raw(record["raw"],case["width"],case["height"])
        control = (bind_point(point,case["controls"],case["controls"],
                              case["width"],case["height"]) if point is not None else None)
        geometry_valid = control is not None
        if geometry_valid != bool(prior_score.get("gate_valid_unique")):
            raise ValueError("H37 point geometric binding changed")
        matched = geometry_valid and unique_exact_label_match(
            case["goal"],control["label"],[entry["label"] for entry in case["controls"]])
        correct = bool(prior_score["target_hit"])
        category = ("correct_retained" if matched and correct else
                    "wrong_retained" if matched else
                    "correct_rejected" if correct else
                    "wrong_unique_rejected" if geometry_valid else
                    "unbound_rejected")
        for group in (counts,langs[case["language"]],layouts[case["layout"]]):
            group[category] += 1
        result_rows.append({"case_id":case["id"],"language":case["language"],
                            "layout":case["layout"],"split":case["split"],
                            "geometry_valid":geometry_valid,"guard_proposed":bool(matched),
                            "target_hit":correct,"category":category})
    if sum(counts.values())!=64:
        raise ValueError("incorrect H38 denominator")
    return {
        "schema":"hala.h38.exact_label_guard_h37_retrospective.v1",
        "h37_model_source_sha":SOURCE_SHA,
        "h37_raw_sha256":RAW_SHA,"h37_audit_sha256":AUDIT_SHA,
        "denominator":64,"counts":dict(sorted(counts.items())),
        "languages":{k:dict(sorted(v.items())) for k,v in langs.items()},
        "layouts":{k:dict(sorted(v.items())) for k,v in layouts.items()},
        "cases":result_rows,
        "model_improved":False,"execution_authorized":False,
        "retrospective_public_authored_synthetic":True,
        "independent_blind_test":False,"native_actions":False,
    }
