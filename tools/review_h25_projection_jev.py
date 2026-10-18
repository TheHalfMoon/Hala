"""Advisory Jev review for Hala H25 V2 source and test deltas only."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "h25/projection_v2.py",
    "tests/test_h25_projection_v2.py",
    "tests/test_h25_projection_pins.py",
    "docs/H25_PROJECTION_ADMISSION_REVIEW.md",
)
QUESTIONS = {
    "answer_leakage_concern": {
        "type": "noul",
        "instructions": "Find a concrete label, synthetic answer bit, partition/rank leak or private expected proposal in H25 V2 model_input. The source_ref envelope is never model input; inspect the actual exported payload.",
        "criteria": {"true": "A reachable reference label or source case rank is included in the model prompt projection.",
                     "false": "Only natural-language goal, neutral UI/observations and legitimate frame equality are public."},
    },
    "source_identity_concern": {
        "type": "noul",
        "instructions": "Find a reachable way for an attacker to substitute a tampered source/family/Arabic goal, bypass immutable SHA256 verification, or silently relabel a DEV row.",
        "criteria": {"true": "A concrete mutating or source bypass path can be admitted.",
                     "false": "Candidate derives from exact pinned canonical frozen H25 source, and full reconstruction rejects change."},
    },
    "unsafe_qualification_concern": {
        "type": "noul",
        "instructions": "Does this code or documentation falsely claim independent Arabic review, held-out TEST performance, RAQM image qualification, native Orcel authorization, or a completed model run?",
        "criteria": {"true": "A specific unperformed scientific/security gate is claimed complete.",
                     "false": "Only synthetic source projection is qualified; GPU, Arabic expert, TEST, native and release remain open."},
    },
}
def main():
    results=[]
    for relative in FILES:
        raw=(ROOT/relative).read_bytes()
        state=("Source-only Hala H25 V2 public input projection, no actual inference, "
               "no real Orcel access. Read exact source:\nFILE="+relative
               +"\nSHA256="+hashlib.sha256(raw).hexdigest()+"\n"
               +raw.decode("utf-8"))
        with tempfile.TemporaryDirectory(prefix="hala-h25-v2-jev-") as tmp:
            request=Path(tmp)/"review.json"
            request.write_text(json.dumps({"state":state,"questions":QUESTIONS},
                                          ensure_ascii=False),encoding="utf-8")
            proc=subprocess.run(["jev","ask",str(request)],capture_output=True,
                                text=True,timeout=120)
        try:
            reply=json.loads(proc.stdout)
        except json.JSONDecodeError:
            reply={"error":"invalid_judge_response","stderr_excerpt":proc.stderr[:250]}
        results.append({"file":relative,"sha256":hashlib.sha256(raw).hexdigest(),
                        "exit_code":proc.returncode,"result":reply})
        print(json.dumps({"file":relative,"exit_code":proc.returncode,
                          "result":reply},ensure_ascii=True),flush=True)
    record={"schema":"hala.h25.projection_v2_jev_advisory.v1",
            "independent_review":False,"model_inference":False,
            "questions":QUESTIONS,"files":results}
    dest=ROOT/"docs"/"reviews"/"h25-projection-v2-jev.json"
    dest.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")
    if any(r["exit_code"]!=0 or r["result"].get("ok") is not True for r in results):
        raise SystemExit("Jev execution failed; never claim review passed")
if __name__=="__main__":
    main()
