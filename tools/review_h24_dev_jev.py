"""Independent-ish Jev typed advisory on *prospective*, DEV-only H24 sources.

This tool cannot substitute for a fresh-context independent human/security review.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "h24/mind_dev.py",
    "tools/build_h24_kaggle.py",
    "tools/verify_h24_dev.py",
    "tests/test_h24_mind_dev.py",
    "tests/test_h24_dev_audit.py",
    "docs/H24_MIND_DEV_BASELINE.md",
    "tools/diagnose_h24_dev.py",
    "tests/test_h24_diagnostics.py",
    "tests/test_h24_protocol_v2.py",
    "docs/H24_MIND_DEV_PROTOCOL_V2.md",
)
QUESTIONS = {
    "label_leakage": {
        "type": "noul",
        "instructions": "Does a concrete path leak H24 expected labels, protected synthetic target tags, or reserved TEST examples to the remote HalaMind model prompt?",
        "criteria": {
            "true": "A reachable reference answer or protected tag enters model input or frozen TEST is loaded for model selection.",
            "false": "Actual prompt projection excludes reference labels and restricted evaluator metadata; only DEV is used."
        }
    },
    "misleading_results": {
        "type": "noul",
        "instructions": "Can a falsified or incomplete downloaded DEV result be accepted by the CPU auditor, or does the source falsely claim a real model run before one occurred?",
        "criteria": {
            "true": "A concrete tampering or missing result is accepted as verified or the text asserts GPU execution without evidence.",
            "false": "Raw outputs are independently rescored with all 20 rows and no real model-quality claim."
        }
    },
    "authority_bypass": {
        "type": "noul",
        "instructions": "Is there a reachable path for local checkpoint download or unauthorized Orcel OS execution through this DEV-only source?",
        "criteria": {
            "true": "An unchecked local model download, OS execution path, or policy grant is present.",
            "false": "Runner checks remote free Kaggle conditions and never executes untrusted GUI actions."
        }
    },
}
def main():
    checked = []
    for name in FILES:
        raw = (ROOT / name).read_bytes()
        state = (
            "Review this Hala H24 DEV-only exploratory baseline; no model GPU run yet. "
            "Orcel remains authority; no model adoption. The source follows.\n"
            + "FILE: " + name
            + "\nSHA256: " + hashlib.sha256(raw).hexdigest()
            + "\n" + raw.decode("utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="hala-h24-jev-") as tmp:
            spec = Path(tmp) / "query.json"
            spec.write_text(json.dumps({"state": state, "questions": QUESTIONS},
                                       ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run(["jev", "ask", str(spec)], capture_output=True,
                                  text=True, timeout=120)
        try:
            review = json.loads(proc.stdout)
        except json.JSONDecodeError:
            review = {"error_type": "invalid_jev_result", "stderr": proc.stderr[:300]}
        item = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                "exit_code": proc.returncode, "result": review}
        checked.append(item)
        print(json.dumps(item, ensure_ascii=True), flush=True)
    evidence = {
        "schema": "hala.h24.dev_jev_advisory.v1",
        "tool": "jev", "independent_pstack_review": False,
        "production_approval": False, "model_gpu_qualified": False,
        "questions": QUESTIONS, "files": checked
    }
    (ROOT / "docs/reviews/h24-dev-jev.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    if any(row["exit_code"] != 0 or row["result"].get("ok") is not True for row in checked):
        raise SystemExit("Actual Jev review failed; do not qualify source")

if __name__ == "__main__":
    main()
