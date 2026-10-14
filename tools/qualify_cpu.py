import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
output = root / 'docs/reviews'
output.mkdir(exist_ok=True)
steps = [
    ('cpu_unittest', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
    ('syntax', [sys.executable, '-m', 'compileall', '-q', 'hala', 'evaluation', 'training', 'data', 'experiments', 'research']),
    ('metadata_integrity', [sys.executable, 'research/validate_sources.py']),
    ('whitespace', ['git', 'diff', '--check']),
    ('sft_integrity', [sys.executable, '-c', "import json;from training.artifacts import load_sft_artifact;from pathlib import Path;p=Path('training/generated-candidate');m=json.loads((p/'manifest.json').read_text());r=load_sft_artifact(p/'examples.jsonl',m);print(json.dumps({'records':len(r),'families':len(m['families']),'heldout_families':len(m['held_out_families']),'arabic_enabled':m['arabic_image_training']['enabled'],'training_executed':False}))"]),
]
results = []
for name, command in steps:
    start = time.perf_counter()
    process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
    log = process.stdout + process.stderr
    path = output / (name + '.log')
    path.write_text(log, encoding='utf-8')
    results.append({'name':name,'command':command,'exit_code':process.returncode,
                    'seconds':time.perf_counter()-start,'log':path.relative_to(root).as_posix(),
                    'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print(json.dumps(results[-1]), flush=True)
report = {'date':'2026-10-14','python':sys.version,'results':results,
          'scope':'CPU infrastructure and helper gates; no foundation-model qualification'}
(output/'cpu-gates-final.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
raise SystemExit(any(row['exit_code'] for row in results))
