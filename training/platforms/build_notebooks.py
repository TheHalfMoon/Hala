"""Derive portable Colab notebooks from canonical scripts; no duplicated implementation."""

import json
import hashlib
from pathlib import Path


def cell(kind, source):
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build(root=None):
    root = root or Path(__file__).resolve().parents[2]
    probe = (root / 'training/scripts/gpu_probe.py').read_text(encoding='utf-8')
    baseline = (root / 'training/scripts/baseline.py').read_text(encoding='utf-8')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'accelerator': 'GPU',
                             'hala_source_sha256': {
                                 'training/scripts/gpu_probe.py': hashlib.sha256(probe.encode('utf-8')).hexdigest(),
                                 'training/scripts/baseline.py': hashlib.sha256(baseline.encode('utf-8')).hexdigest()}},
                'cells': [cell('markdown', '# Hala unchanged-model diagnostic\nUse a free GPU only. No paid infrastructure authorized. Weights remain on the remote notebook, never on the personal computer. Eight geometric checks do not qualify complete-task performance.\n'),
                          cell('code', probe),
                          cell('markdown', 'Continue only after the probe reports a real GPU. Install the pinned official processor runtime; do not activate Colab plans, AI assistants, hosting or customer-facing services.'),
                          cell('code', '%pip install --quiet transformers==5.16.1\n'),
                          cell('code', 'import os\nos.environ["HALA_REMOTE_BASELINE_ALLOWED"] = "1"\n' + baseline)]}
    destination = root / 'training/platforms/colab-baseline.ipynb'
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding='utf-8')
    return destination


if __name__ == '__main__':
    print(build())
