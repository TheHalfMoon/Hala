"""Advisory Jev semantic review on current Orcel naming/contract compatibility docs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
docs = [ROOT / "README.md", ROOT / "docs/ORCEL_INTEGRATION.md"]
state = "\n".join(p.name + " SHA256=" + hashlib.sha256(p.read_bytes()).hexdigest() + "\n" + p.read_text(encoding="utf-8") for p in docs)
questions = {
    "contract_integrity": {
        "type": "noul",
        "instructions": "Does the Orcel compatibility document concretely claim that the current Hala and Orcel schemas or execution authority are production-integrated, contrary to the pinned evidence?",
        "criteria": {
            "true": "The source asserts actual integration or grants AI output Orcel execution authority, or falsely equates different request schemas.",
            "false": "The source keeps all integration hypothetical, all AI output untrusted, and requires a versioned trusted adapter."
        }
    }
}
with tempfile.TemporaryDirectory(prefix="hala-orcel-review-") as temp:
    request = Path(temp) / "request.json"
    request.write_text(json.dumps({"state": state, "questions": questions}), encoding="utf-8")
    response = subprocess.run(["jev", "ask", str(request)], capture_output=True, text=True, timeout=90)
result = {"tool": "jev ask", "type": "advisory", "source_files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in docs},
          "exit_code": response.returncode, "response": None, "error": None,
          "independent_pstack_judge": False}
try:
    result["response"] = json.loads(response.stdout)
except json.JSONDecodeError:
    result["error"] = response.stderr[:800]
target = ROOT / "docs/orcel-compat-jev-result.json"
target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"exit_code": response.returncode, "response": result["response"], "error": result["error"]}, sort_keys=True))
raise SystemExit(response.returncode)
