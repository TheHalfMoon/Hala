"""Capture genuine bounded Jev typed judgments for the H61 sources.

The scores are advisory. They never approve a release, they never replace the
genuine tests and they never turn a measured number into a qualification. The
file hashes are recorded so a reviewer can tell whether the judged source is
the source that is actually on the branch.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = [
    'h61/__init__.py',
    'h61/source_export.py',
    'h61/curriculum.py',
    'h61/prompt_contract.py',
    'h61/supervision.py',
    'h61/baseline.py',
    'h61/official_eval.py',
    'h61/official_eval_run.py',
    'h61/colab_run.py',
    'h61/remote_entry.py',
    'h61/source_qualification.py',
    'h61/update_catalog.py',
    'training/preflight.py',
    'training/scripts/train_adapter.py',
    'tests/test_h61_curriculum.py',
    'tests/test_h61_official_eval.py',
    'tests/test_h61_supervision.py',
]
questions = {
    'correctness_concern': {
        'type': 'noul',
        'instructions': 'Is an evident correctness defect present in this source under the stated scope?',
        'criteria': {
            'true': 'Concrete reachable error, violated invariant or falsely claimed measurement.',
            'false': 'No evident defect; unknown runtime support alone does not prove a defect.',
        },
    },
    'evidence_concern': {
        'type': 'noul',
        'instructions': 'Does this source falsely treat mock, planned, advertised or synthetic measurements as a real trained-model or benchmark qualification?',
        'criteria': {
            'true': 'False model, training, benchmark or deployment qualification is claimed.',
            'false': 'Mock, planned or unqualified status is explicit; the source only claims what it measured.',
        },
    },
    'leakage_concern': {
        'type': 'noul',
        'instructions': 'Does this source read an evaluation, benchmark or test answer into a training row or into a model prompt, or rewrite an original source label?',
        'criteria': {
            'true': 'An evident path leaks an answer into training or into a prompt, or rewrites an original label.',
            'false': 'The source separates training from evaluation and preserves the original source records.',
        },
    },
}
temp = root.parent / 'jev-h61-inputs'
temp.mkdir(exist_ok=True)
results = []
for relative in files:
    data = (root / relative).read_bytes()
    source = data.decode('utf-8')
    state = (
        'Scope: Hala H61 original-corpus ingest, capability-balanced curriculum, '
        'coordinate-focused supervision, official upstream public evaluation and '
        'the real remote Colab run. The model is untrusted advice and never a '
        'completion receipt. The upstream official evaluation split is never a '
        'training source. Real model runs happen only on an authorised free Colab '
        'Linux CUDA session and no Kaggle GPU session is used. '
        'File: ' + relative + '\n' + source
    )
    request = temp / (relative.replace('/', '--') + '.json')
    request.write_text(json.dumps({'state': state, 'questions': questions}), encoding='utf-8')
    started = time.perf_counter()
    process = subprocess.run(['jev', 'ask', str(request)], capture_output=True, text=True,
                             timeout=180)
    row = {
        'file': relative,
        'source_sha256': hashlib.sha256(data).hexdigest(),
        'exit_code': process.returncode,
        'wall_seconds': time.perf_counter() - started,
    }
    try:
        row['response'] = json.loads(process.stdout)
    except ValueError:
        row['error'] = process.stderr[:2000]
    results.append(row)
    print(json.dumps({'file': relative, 'exit_code': row['exit_code']}), flush=True)
result = {
    'tool': 'jev 2026.919.0',
    'scope': ('file-level bounded typed semantic review, advisory; not independent '
              'release approval and not a qualification'),
    'date': time.strftime('%Y-%m-%d'),
    'questions': questions,
    'results': results,
    'billing': ('API responses report tokens, not billing. No billing, top-up or '
                'subscription action was performed.'),
}
destination = root / 'docs/reviews/h61-jev-judgments.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(f'H61_JEV_REVIEW {sum(1 for row in results if row["exit_code"] == 0)}/{len(results)}')
if any(row['exit_code'] != 0 for row in results):
    raise SystemExit(1)
