"""Actual Jev typed advisory judgments; not an independent release reviewer."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
FILES=("h24/data.py","tests/test_h24_data.py","tests/test_h24_archive.py",
       "docs/H24_PROSPECTIVE_BILINGUAL.md")
QUESTIONS={
    "integrity_concern":{
        "type":"noul",
        "instructions":"Is a concrete reachable implementation defect evident: family/split leakage, source identity tampering, historical H19 freeze violation, wrong test expectations or loss of provenance?",
        "criteria":{"true":"Source permits a specific unaudited row, family, label, history or rights substitution.",
                    "false":"The reviewed code and tests enforce deterministic regenerated source and isolated train/dev/test splits."}},
    "unsafe_training_concern":{
        "type":"noul",
        "instructions":"Could the synthetic generator or exported training rows concretely teach Hala to obey hostile page instructions, take unauthorized destructive actions, or treat synthetic untrusted observations as production authority?",
        "criteria":{"true":"A concrete labeled output authorizes hostile text or a protected/unsafe action.",
                    "false":"Safe requests ignore untrusted page instructions, unsafe user requests abstain, all output remains untrusted."}},
    "misleading_qualification_concern":{
        "type":"noul",
        "instructions":"Do source or docs concretely claim visually qualified Arabic screenshot training, native Orcel integration, completed model performance, or independent test quality that was not demonstrated?",
        "criteria":{"true":"Source or documentation claims a specific experiment, integration or visual qualification that did not occur.",
                    "false":"Candidate source is text-only, originally synthetic and unqualified pending independent evaluation."}}
}
def main():
    results=[]
    for file in FILES:
        payload=(ROOT/file).read_bytes()
        state=("Hala H24 prospective synthetic bilingual data, source-only. No GPU model or Orcel integration, no Type images or RAQM rendering. Historical H22 tests must remain untouched.\n"+
               f"FILE={file}\nSHA256={hashlib.sha256(payload).hexdigest()}\nSOURCE:\n"+payload.decode("utf-8"))
        with tempfile.TemporaryDirectory(prefix="hala-h24-jev-") as temp:
            spec=Path(temp)/"review.json"
            spec.write_text(json.dumps({"state":state,"questions":QUESTIONS},ensure_ascii=False),encoding="utf-8")
            p=subprocess.run(["jev","ask",str(spec)],capture_output=True,text=True,timeout=120)
        try:
            reply=json.loads(p.stdout)
        except json.JSONDecodeError:
            reply={"error_type":"invalid_jev_response","stderr_excerpt":p.stderr[:300]}
        row={"file":file,"sha256":hashlib.sha256(payload).hexdigest(),
             "exit_code":p.returncode,"result":reply}
        print(json.dumps(row,ensure_ascii=True),flush=True)
        results.append(row)
    output=ROOT/"docs/reviews/h24-jev.json"
    output.write_text(json.dumps({"schema":"hala.h24.jev_advisory.v1",
        "not_independent_pstack":True,"not_production_approval":True,
        "questions":QUESTIONS,"files":results},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if any(x["exit_code"]!=0 or x["result"].get("ok") is not True for x in results):
        raise SystemExit("Genuine Jev review not successful")

if __name__=="__main__":
    main()
