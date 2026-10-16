"""Qualify a newly restored test-restart runtime; no model load, fetch, checkout or prewarm."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/kaggle/working/hala-h21')
COMMIT = '07591f109b6683c9302fcf890259ddf0bde91494'
FREEZE = '98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b'
DATA = 'ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e'
ADAPTER = '7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d'
IDENTITY = {'base_model': 'Qwen/Qwen3-VL-2B-Instruct', 'base_revision': '89644892e4d85e24eaac8bacfd4f463576704203',
    'data_sha256': 'af540e4db2289540fb8adecd0665439c4d6055ffc86673a3286e902c86c0340e',
    'evaluation_sha256': 'e9f98694089d5411d3a6c570c7de2ede8f82cbeb3132e47904d17301030c0768',
    'source_commit': '5b6eb4bf29acbaf21aa199f113f78af4f268b3b7'}


def clean_source(root):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root / 'source', text=True, timeout=30).strip()
    status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root / 'source',
                            capture_output=True, text=True, check=True, timeout=30).stdout
    if actual != COMMIT or status.strip():
        raise ValueError('Exact complete clean source required before qualification imports')


def qualify(expected_gpu_uuid, root=ROOT):
    clean_source(root)
    output = root / 'h22-test-restart-qualification.json'
    if output.exists():
        raise FileExistsError('Qualification evidence already exists')
    sys.path.insert(0, str(root / 'source'))
    os.environ['HALA_REMOTE_DEVELOPMENT_ALLOWED'] = '1'
    from evaluation import development_precision as precision
    from evaluation.development_runtime import evaluation_guard, VERSIONS
    from evaluation.development_adapter import verified_adapter
    from development.freeze import verify_bundle
    from development.media import FONT_SHA256
    provider = evaluation_guard('kaggle')
    precision.verify_sources(FREEZE)
    precision.check_cache('/tmp/hala-h22-cache')
    for key in precision.CACHE_KEYS:
        os.environ[key] = '/tmp/hala-h22-cache'
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    manifest = verify_bundle(root / 'frozen', require_media=True, expected_manifest_sha256=DATA)
    with verified_adapter(root / 'retained-adapter', IDENTITY, ADAPTER):
        pass
    font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if hashlib.sha256(font.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('Qualified immutable font required')
    expected = VERSIONS | precision.PINS | {'peft': '0.20.0', 'safetensors': '0.8.0', 'torch': '2.11.0+cu128'}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected or importlib.util.find_spec('torchao') is not None:
        raise ValueError('Exact original pins and absent interfering optional TorchAO required')
    clean_source(root)
    import torch
    import bitsandbytes
    import accelerate
    import peft
    import safetensors
    import transformers
    from PIL import features, __version__ as pillow_version
    imported = {'torch': torch.__version__, 'bitsandbytes': bitsandbytes.__version__, 'accelerate': accelerate.__version__,
        'peft': peft.__version__, 'safetensors': safetensors.__version__, 'transformers': transformers.__version__, 'Pillow': pillow_version}
    if imported != expected or not torch.cuda.is_available() or features.version_feature('raqm') != '0.10.1':
        raise ValueError('Actual imports/CUDA/RAQM do not match fixed study')
    uuid = precision.resolve_device_uuid(torch)
    if uuid != expected_gpu_uuid:
        raise ValueError('New test-restart GPU differs from externally observed device UUID')
    sample = precision.process_sample(uuid)
    if sample['device'] is None:
        raise ValueError('Actual UUID-bound device sampler unavailable')
    sampler = precision.CostSampler(limit=1)
    sampler.device_uuid = uuid
    sampler.start()
    samples = sampler.stop()
    if not samples['samples'] or samples['samples'][0].get('device') is None:
        raise ValueError('Actual bounded sampler unavailable')
    report = {'schema': 'hala.h22.remote_qualification.v1', 'attempt': 3, 'source_commit': COMMIT,
        'source_freeze_sha256': FREEZE, 'h19_manifest_sha256': DATA, 'h19_image_count': manifest['image_count'],
        'provider': provider, 'packages': versions, 'imported_packages': imported,
        'gpu': torch.cuda.get_device_name(0), 'gpu_uuid': uuid, 'cuda': torch.version.cuda,
        'gpu_capability': list(torch.cuda.get_device_capability(0)), 'raqm': features.version_feature('raqm'),
        'font_sha256': FONT_SHA256, 'device_sample': sample, 'bounded_sampler': samples,
        'unique_physical_cache_bytes': precision.disk_bytes('/tmp/hala-h22-cache'),
        'cache_state': 'Observed restored cache before any test-restart prewarm; prewarm must be recorded separately',
        'cache_prewarm_performed': False, 'adapter_manifest_sha256': ADAPTER,
        'model_loaded': False, 'prior_dev_results_retained': True, 'remaining_test_results_retrieved': False, 'actual_precision_quality_qualified': False}
    temporary = output.with_suffix('.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    print('HALA_H22_TEST_RESTART_QUALIFICATION:' + json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-gpu-uuid', required=True)
    qualify(**vars(parser.parse_args()))
