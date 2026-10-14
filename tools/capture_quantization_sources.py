"""Authoring evidence lookup: registry/source text only, no imports or model weights."""
import hashlib
import json
import urllib.request
from evaluation.heldout_fixtures import ROOT, canonical, digest

SOURCES = (
    'https://pypi.org/pypi/bitsandbytes/0.50.2/json',
    'https://pypi.org/pypi/accelerate/1.12.0/json',
    'https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/utils/import_utils.py',
    'https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/quantizers/quantizer_bnb_4bit.py',
    'https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/quantizers/quantizer_bnb_8bit.py',
    'https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.50.2/README.md',
)


def capture():
    records = []
    for url in SOURCES:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        text = raw.decode('utf-8')
        if url.endswith('/json'):
            data = json.loads(text)
            details = {key: data['info'][key] for key in
                       ('version', 'requires_python', 'requires_dist')}
            details['linux_wheels'] = [f['filename'] for f in data['urls']
                                       if 'manylinux' in f['filename']]
        else:
            details = {'source_text': text}
        records.append({'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'details': details})
    target = ROOT / 'experiments/quantization'
    (target / 'official-sources.json').write_bytes(canonical({
        'schema': 'hala.h17.official-source-evidence.v1',
        'lookup_date': '2026-10-14', 'gpu_compatibility_qualified': False, 'sources': records}))
    baseline_path = 'experiments/quantization/baseline-fp16.json'
    baseline = json.loads((ROOT / baseline_path).read_text(encoding='utf-8'))
    (target / 'config.json').write_bytes(canonical({
        'schema': 'hala.h17.config.v1', 'modes': ['int8', 'nf4'],
        'model_revision': baseline['model_revision'], 'h14_manifest_sha256':
        baseline['manifest_sha256'], 'baseline_path': baseline_path,
        'baseline_sha256': digest(ROOT / baseline_path),
        'packages': {**baseline['packages'], 'bitsandbytes': '0.50.2', 'accelerate': '1.12.0'},
        'seed': 42, 'attention': 'sdpa', 'max_new_tokens': 48, 'max_time_seconds': 15,
        'deadline_seconds': 1800, 'adoption_qualified': False}))


if __name__ == '__main__':
    capture()
