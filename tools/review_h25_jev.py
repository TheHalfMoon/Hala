"""Actual Jev semantic-advisory review of H25 first-party bilingual corpus."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    "h25/corpus.py",
    "tests/test_h25_corpus.py",
    "tests/test_h25_frozen.py",
    "docs/H25_BILINGUAL_SEMANTIC_CANDIDATE.md",
)
QUESTIONS={
    "label_integrity":{
        "type":"noul",
        "instructions":"Does the prospective H25 frozen dataset contain a concrete source/case/label mismatch, cross-family leakage, fabricated Arabic correctness, or a path to silently alter expected task outcomes?",
        "criteria":{"true":"A specific bad row, wrong expected output or unvalidated corruption can be admitted.",
                    "false":"Regeneration and source tests bind all case states, labels, families, languages and file hashes."}},
    "model_input_leakage":{
        "type":"noul",
        "instructions":"Can actual H25 DEV model_input records expose private answer/oracle labels or source_ref IDs as visible model instructions, or can a page-injected instruction be promoted to permission?",
        "criteria":{"true":"A concrete exposed private oracle reference or authority escalation is present.",
                    "false":"The public DEV projection separates tracking from model input and keeps page text untrusted."}},
    "false_release_qualification":{
        "type":"noul",
        "instructions":"Do code or docs falsely claim Hala training, Arabic visual/RAQM qualification, new TEST independence, deployment, production rights approval or native Orcel actions?",
        "criteria":{"true":"A concrete unperformed real-model validation or legally unqualified production claim appears.",
                    "false":"Research dataset is labeled synthetic, not visually trained or safety certified, and rights require review."}},
}
def main():
    results=[]
    for file in FILES:
        data=(ROOT/file).read_bytes()
        state=(
            "Hala H25 new synthetic authored bilingual corpus only; "
            "no new TEST/model inference, no Orcel operations, no GPU. "
            "Report reachable concrete source defects, not invented features. "
            "\nFILE="+file+"\nSHA256="+hashlib.sha256(data).hexdigest()
            +"\nSOURCE:\n"+data.decode("utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="hala-h25-jev-") as tmp:
            request=Path(tmp)/"request.json"
            request.write_text(json.dumps({"state":state,"questions":QUESTIONS},
                                          ensure_ascii=False),encoding="utf-8")
            proc=subprocess.run(["jev","ask",str(request)],capture_output=True,
                                text=True,timeout=120)
        try:
            answer=json.loads(proc.stdout)
        except json.JSONDecodeError:
            answer={"error_type":"invalid_review_reply","stderr_excerpt":proc.stderr[:350]}
        result={"file":file,"sha256":hashlib.sha256(data).hexdigest(),
                "exit_code":proc.returncode,"result":answer}
        print(json.dumps({"file":file,"exit_code":proc.returncode,"result":answer},
                         ensure_ascii=True),flush=True)
        results.append(result)
    document={
        "schema":"hala.h25.jev_advisory.v1",
        "source_only":True,
        "independent_pstack_approval":False,
        "model_training_performed":False,
        "native_orcel_executed":False,
        "questions":QUESTIONS,
        "files":results,
    }
    (ROOT/"docs/reviews/h25-jev.json").write_text(
        json.dumps(document,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8")
    if any(r["exit_code"]!=0 or r["result"].get("ok") is not True for r in results):
        raise SystemExit("Jev advisory review did not complete; do not publish as approved")


if __name__=="__main__":
    main()
