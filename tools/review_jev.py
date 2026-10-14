"""Capture genuine bounded Jev typed judgments; no gate approves from these scores."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = ['hala/contracts/validation.py', 'hala/contracts/request.py',
         'hala/contracts/proposal.py', 'hala/contracts/responses.py', 'hala/contracts/mock.py',
         'hala/serving/__init__.py', 'hala/inference/__init__.py',
         'data/factory.py', 'data/export.py', 'evaluation/engine.py',
         'evaluation/metrics.py', 'experiments/cost/calculator.py',
         'training/preflight.py', 'training/scripts/train_adapter.py',
         'training/scripts/baseline.py', 'training/artifacts.py', 'training/checkpoints.py', 'training/masking.py', 'training/scripts/convert_sft.py', 'tools/review_jev.py', 'tools/qualify_cpu.py']
questions = {
    'correctness_concern': {'type': 'noul', 'instructions': 'Is an evident correctness defect present in this source under the stated scope?',
                           'criteria': {'true': 'Concrete reachable error, violated invariant or falsely claimed measurement.', 'false': 'No evident defect; unknown runtime support alone does not prove a defect.'}},
    'security_concern': {'type': 'noul', 'instructions': 'Does this source evidently grant model authority, trust false completion, permit cross-tenant leakage, arbitrary execution or unbounded attacker work?',
                        'criteria': {'true': 'An evident path violates one of those invariants.', 'false': 'The source provides bounded untrusted advice or offline synthetic analysis without that path.'}},
    'evidence_concern': {'type': 'noul', 'instructions': 'Does this source falsely treat mock or synthetic measurements as real trained-model or production qualification?',
                        'criteria': {'true': 'False model/training/deployment qualification is claimed.', 'false': 'Mock, scenario or unqualified status is explicit; actual model scripts are merely candidate code.'}}
}
temp = root.parent / 'jev-inputs'
temp.mkdir(exist_ok=True)
results = []
for relative in files:
    data = (root / relative).read_bytes()
    source = data.decode('utf-8')
    state = ('Scope: Hala CPU mock and independent synthetic harness, no actual execution authority. '
             'Aladdin local authority is final. Real model scripts unexecuted/unqualified and remote-only. '
             'Cost inputs are scenarios. Other strict contract boundaries may be in imported modules. '
             'File: ' + relative + '\n' + source)
    request = temp / (relative.replace('/', '--') + '.json')
    request.write_text(json.dumps({'state': state, 'questions': questions}), encoding='utf-8')
    started = time.perf_counter()
    process = subprocess.run(['jev', 'ask', str(request)], capture_output=True, text=True, timeout=60)
    row = {'file': relative, 'source_sha256': hashlib.sha256(data).hexdigest(),
           'exit_code': process.returncode, 'wall_seconds': time.perf_counter() - started}
    try:
        row['response'] = json.loads(process.stdout)
    except ValueError:
        row['error'] = process.stderr[:2000]
    results.append(row)
    print(json.dumps(row), flush=True)
result = {'tool': 'jev 2026.919.0', 'scope': 'file-level bounded typed semantic review, advisory; not independent release approval',
          'date': '2026-10-14', 'questions': questions, 'results': results,
          'billing': 'API responses report tokens, not billing. No billing/top-up/subscription action performed.'}
destination = root / 'docs/reviews/jev-source-judgments.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
if any(row['exit_code'] != 0 for row in results):
    raise SystemExit(1)
