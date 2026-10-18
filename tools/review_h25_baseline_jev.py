"""Genuine typed Jev advisory on H25 DEV-only source; not a PStack signoff."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    "h25/baseline.py",
    "tools/verify_h25_dev.py",
    "tests/test_h25_baseline.py",
    "tests/test_h25_audit.py",
    "docs/H25_BASELINE_PROTOCOL.md",
    "docs/QWENGYRE_RESEARCH_ADR.md",
    "tools/build_h25_kaggle.py",
    "tests/test_h25_kaggle_builder.py",
)
QUESTIONS={
    "model_input_label_leak":{
        "type":"noul",
        "instructions":"Find an actual path that sends H25 source_ref, oracle_private, H24/H22 TEST cases, supervised labels or action authorization into the 42 DEV-only model payload.",
        "criteria":{"true":"Concrete reachable reference leakage, held-out TEST reuse or native authorization occurs.",
                    "false":"Only the public model_input reaches HalaMind and all outputs remain untrusted."}
    },
    "audit_accepts_forgery":{
        "type":"noul",
        "instructions":"Identify a reachable falsified/partial downloaded H25 raw result which the independent CPU replay falsely considers a complete same-model/same-source result.",
        "criteria":{"true":"Specific forged data, duplicate case or changed source/model state can pass.",
                    "false":"Exact source/model/protocol, all 42 raw rows, external digest and score replay required."}
    },
    "false_scientific_claim":{
        "type":"noul",
        "instructions":"Check whether source/documentation falsely implies H25 Kaggle GPU measured, QwenGyre RL performed, independent holdout Arabic/Type correctness, published Gyre code licensed, or native Orcel use.",
        "criteria":{"true":"An unperformed experiment/license/safety certification is stated as done.",
                    "false":"Only local CPU source preparation is claimed, with research limitations exposed."}
    },
}
def main():
    out=[]
    for file in FILES:
        data=(ROOT/file).read_bytes()
        state=("Only Hala H25 prospective 42 DEV source, no Kaggle GPU run, no Orcel changes. "
               "Review exact source bytes, report concrete actionable defects.\nFILE: "+file+
               "\nSHA256: "+hashlib.sha256(data).hexdigest()+"\n"+
               data.decode("utf-8"))
        with tempfile.TemporaryDirectory(prefix="hala-h25-baseline-jev-") as tmp:
            spec=Path(tmp)/"review.json"
            spec.write_text(json.dumps({"state":state,"questions":QUESTIONS},
                                       ensure_ascii=False),encoding="utf-8")
            proc=subprocess.run(["jev","ask",str(spec)],capture_output=True,
                                text=True,timeout=120)
        try:
            verdict=json.loads(proc.stdout)
        except json.JSONDecodeError:
            verdict={"error":"invalid_jev_reply","stderr_excerpt":proc.stderr[:350]}
        record={"file":file,"sha256":hashlib.sha256(data).hexdigest(),
                "exit_code":proc.returncode,"result":verdict}
        out.append(record)
        print(json.dumps({"file":file,"exit_code":proc.returncode,
                          "result":verdict},ensure_ascii=True),flush=True)
    result={
        "schema":"hala.h25.dev_baseline_jev_advisory.v1",
        "actual_gpu_results":False,
        "independent_security_approval":False,
        "questions":QUESTIONS,"files":out,
    }
    folder=ROOT/"docs"/"reviews"
    (folder/"h25-baseline-jev.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8")
    if any(item["exit_code"]!=0 or item["result"].get("ok") is not True for item in out):
        raise SystemExit("Jev source review failed")

if __name__=="__main__":
    main()
