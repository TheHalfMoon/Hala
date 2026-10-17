"""Genuine scoped Jev judgments for the H22 batch-preparation source.

Advisory semantic judgments are NOT a security release or PStack substitute.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = ("tools/h22_batch_operator.py", "tools/generate_h22_batch.py",
         "tests/test_h22_batch_operator.py")
QUESTIONS = {
    "correctness_concern": {"type": "noul",
        "instructions": "Is a concrete reachable correctness defect, inconsistent hardware/data identity, or evidence retention bug evident in this source?",
        "criteria": {"true": "A demonstrated design path can mismatch results, overwrite evidence, or misattribute a batch.",
                     "false": "No evident reachable defect in this file's stated bounded scope."}},
    "security_concern": {"type": "noul",
        "instructions": "Does the implementation evidently bypass environment, source hashes, safe ZIP extraction, authentication, authority, or spending controls?",
        "criteria": {"true": "A concrete attacker-controlled input or authority path violates one of the named protections.",
                     "false": "Source contains bounded synthetic evaluation with reviewed external trust checks."}},
    "evidence_concern": {"type": "noul",
        "instructions": "Does the source conflate preflight, experimental batch outputs, real saved-version public retention, or release qualification?",
        "criteria": {"true": "It claims durable/qualified outcomes before immutable retrieval and independent audit.",
                     "false": "It distinguishes prepared, executed, retrieved, audited, and adopted states."}}
}
RESULTS = ROOT / "docs/reviews/h22-batch-jev.json"


def main():
    temp = ROOT.parent / "jev-inputs" / "h22-batch-retention"
    temp.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        raw = (ROOT / name).read_bytes()
        state = "Scope: new H22 TEST batch recovery. Original source 07591, H19 data and H21 adapter unchanged. Earlier FP16 retained, lost INT8 unknown, NF4 never started. New Kaggle Save & Run All source only; NO actual remote execution, no adoption, no pricing claim. File: " + name + "\n" + raw.decode("utf-8")
        request = temp / (Path(name).name + ".json")
        request.write_text(json.dumps({"state": state, "questions": QUESTIONS}), encoding="utf-8")
        started = time.perf_counter()
        result = subprocess.run(["jev", "ask", str(request)], capture_output=True, text=True, timeout=90)
        row = {"file": name, "sha256": hashlib.sha256(raw).hexdigest(),
               "exit_code": result.returncode, "seconds": round(time.perf_counter() - started, 4)}
        try:
            row["response"] = json.loads(result.stdout)
        except json.JSONDecodeError:
            row["error"] = result.stderr[:1000]
        rows.append(row)
        print(json.dumps(row), flush=True)
    RESULTS.write_text(json.dumps({"tool": "Jev genuine CLI", "purpose": "advisory semantic review",
        "billing": "Existing account; invoice/cost unverified; no credit purchase",
        "pstack_status": "UNAVAILABLE; independent PStack gate remains blocked",
        "questions": QUESTIONS, "files": rows}, indent=2) + "\n", encoding="utf-8")
    if any(x["exit_code"] != 0 for x in rows):
        raise SystemExit("One or more Jev calls failed; preserve review gate")
    return rows


if __name__ == "__main__":
    main()
