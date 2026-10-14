"""Authoring-only freeze. Never invoked automatically by the model runner."""
import json
from evaluation.heldout_fixtures import ROOT, canonical, digest

FILES = ('evaluation/quantization.py', 'evaluation/quantization_runner.py',
         'experiments/quantization/config.json', 'experiments/quantization/baseline-fp16.json',
         'experiments/quantization/official-sources.json', 'docs/H17_QUANTIZATION.md',
         'specs/H17-paired-quantization.md', 'tests/test_quantization.py',
         'tools/freeze_quantization.py', 'tools/capture_quantization_sources.py')


def freeze():
    payload = {'schema': 'hala.h17.source-freeze.v1', 'phase': 'before_quantized_model_runs',
               'model_runs': 0, 'files': {name: digest(ROOT / name) for name in FILES}}
    (ROOT / 'experiments/quantization/source-freeze.json').write_bytes(canonical(payload))


if __name__ == '__main__':
    freeze()
