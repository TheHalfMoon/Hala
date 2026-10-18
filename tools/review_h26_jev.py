"""Hala H26 local typed Jev advisory; not independent PStack/security approval."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    "h26/choice.py","h26/corpus.py","h26/runner.py",
    "tools/build_h26_kaggle.py","tools/verify_h26_dev.py",
    "tests/test_h26_choice.py","tests/test_h26_corpus.py",
    "tests/test_h26_runner.py","tests/test_h26_kaggle_audit.py",
    "docs/H26_CHOICE_DEV_PROTOCOL.md",
)
QUESTIONS={
    "unsafe_output_admission":{
        "type":"noul",
        "instructions":"Does an H26 model-controlled code or malformed old H25 JSON get executed, gain native Orcel authority, or bypass the strict C0..C7 untrusted-advisory schema? Cite a concrete reachable source path.",
        "criteria":{"true":"Reachable native action or permission escalation/incorrect output accepted.",
                    "false":"Only source-checked exact code maps to untrusted typed proposal or withheld None; no native executor."},
    },
    "source_label_or_scoring":{
        "type":"noul",
        "instructions":"Identify a concrete H26 model prompt leakage of private expected labels, reused H22/H24 TEST, or a way a forged 42-case result passes exact source hash and independent scoring checks.",
        "criteria":{"true":"Specific reference/split/forged score can be accepted as legitimate.",
                    "false":"Model sees public input only; scorer and raw auditor bind complete cases and chosen code to separate oracle."},
    },
    "false_qualification":{
        "type":"noul",
        "instructions":"Do source/docs falsely claim actual Kaggle H26 model evaluation, independent Arabic semantics, real GUI success, validated commercial rights, independent security review, or QwenGyre training?",
        "criteria":{"true":"A stated completed external experiment or accreditation is missing.",
                    "false":"Only synthetic CPU/source research readiness claimed, all external gates open."},
    },
}
def main():
    notes=[]
    for rel in FILES:
        data=(ROOT/rel).read_bytes()
        state=("Hala H26 prospective research code only; NO model/GPU measurement yet, no Orcel edits. "
               "Inspect this exact file for concrete defects; do not invent evidence.\nFILE="+rel+
               "\nSHA256="+hashlib.sha256(data).hexdigest()+"\nSOURCE:\n"+data.decode("utf8"))
        with tempfile.TemporaryDirectory(prefix="hala-h26-jev-") as tmp:
            req=Path(tmp)/"questions.json"
            req.write_text(json.dumps({"state":state,"questions":QUESTIONS},ensure_ascii=False),encoding="utf8")
            result=subprocess.run(["jev","ask",str(req)],capture_output=True,text=True,timeout=120)
        try:response=json.loads(result.stdout)
        except json.JSONDecodeError:response={"error":"unparseable_judge_reply","stderr_excerpt":result.stderr[:350]}
        entry={"file":rel,"sha256":hashlib.sha256(data).hexdigest(),
               "exit_code":result.returncode,"result":response}
        notes.append(entry)
        print(json.dumps({"file":rel,"exit_code":result.returncode,"result":response},
                         ensure_ascii=True),flush=True)
    evidence={
        "schema":"hala.h26.jev_typed_advisory.v1",
        "independent_security_signoff":False,"model_inference_performed":False,
        "source_only":True,"questions":QUESTIONS,"files":notes,
    }
    folder=ROOT/"docs"/"reviews"
    (folder/"h26-jev.json").write_text(json.dumps(evidence,sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    if any(note["exit_code"]!=0 or note["result"].get("ok") is not True for note in notes):
        raise SystemExit("H26 Jev did not pass as an advisory; no publication")


if __name__=="__main__":
    main()
