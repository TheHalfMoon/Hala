"""H17 experimental comparison helpers; CPU checks never qualify quantized inference."""
import json
import os
from pathlib import Path
import platform
import statistics

from evaluation.heldout_fixtures import ROOT, digest, verify_freeze

CONFIG = ROOT / 'experiments/quantization/config.json'
FREEZE = ROOT / 'experiments/quantization/source-freeze.json'
CACHE = '/tmp/hala-h17-cache'


def quantization_config(mode):
    if mode == 'int8':
        return {'load_in_8bit': True, 'llm_int8_threshold': 6.0,
                'llm_int8_enable_fp32_cpu_offload': False}
    if mode == 'nf4':
        return {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_quant_storage': 'uint8', 'llm_int8_enable_fp32_cpu_offload': False}
    raise ValueError('Only prespecified int8 or nf4 experiments allowed')


def remote_guard():
    if (platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_QUANT_ALLOWED') != '1'
            or not os.environ.get('COLAB_RELEASE_TAG')):
        raise ValueError('Explicit opt-in Linux Colab required; local weights forbidden')
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or Path('/kaggle').exists():
        raise ValueError('Kaggle blocked')
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
                'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME'):
        value = os.environ.get(key)
        if value and not Path(value).resolve().is_relative_to(Path(CACHE).resolve()):
            raise ValueError('Only ephemeral H17 cache permitted: ' + key)
    if Path(CACHE).is_symlink():
        raise ValueError('Cache symlink forbidden')


def verify_sources(expected_sha256=None):
    if expected_sha256 is not None and digest(FREEZE) != expected_sha256:
        raise ValueError('Externally pinned H17 source freeze mismatch')
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    for relative, expected in frozen['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('H17 source/config/protocol changed: ' + relative)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    dataset = verify_freeze(config['h14_manifest_sha256'])
    baseline = json.loads((ROOT / config['baseline_path']).read_text(encoding='utf-8'))
    if digest(ROOT / config['baseline_path']) != config['baseline_sha256']:
        raise ValueError('Actual baseline changed')
    if (baseline.get('status') != 'tested' or len(baseline['cases']) != 64
            or baseline['manifest_sha256'] != config['h14_manifest_sha256']
            or baseline['model_revision'] != config['model_revision']
            or baseline['precision'] != 'fp16' or baseline['attention'] != 'sdpa'):
        raise ValueError('Actual frozen complete FP16 baseline required')
    if [r['id'] for r in baseline['cases']] != [c['id'] for c in dataset['cases']]:
        raise ValueError('Baseline task mapping differs')
    return config, dataset, baseline


def verify_media(baseline, image_dir):
    images = baseline['render_provenance']['images']
    rows = baseline['cases']
    if len(images) != len(rows) or len({r['id'] for r in images}) != len(images):
        raise ValueError('Complete unique media mapping required')
    paths = []
    for row, media in zip(rows, images):
        name = media['path']
        path = image_dir / name
        if (Path(name).name != name or not name.endswith('.png') or
                media['id'] != row['id'] or media['png_sha256'] != row['image_sha256'] or
                not path.is_file() or path.is_symlink() or digest(path) != media['png_sha256']):
            raise ValueError('Frozen paired media mismatch')
        paths.append(path)
    return paths


def paired(baseline, candidate):
    def index(rows):
        result = {row['id']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate case')
        return result
    base, quant = index(baseline), index(candidate)
    if not base or base.keys() != quant.keys():
        raise ValueError('Full paired case set required')
    differences = []
    for key, before in base.items():
        after = quant[key]
        if any(before[field] != after[field] for field in ('image_sha256', 'family', 'language')):
            raise ValueError('Paired media/task mismatch')
        differences.append({'id': key, 'family': before['family'], 'language': before['language'],
                            'baseline_hit': before['hit'], 'quantized_hit': after['hit'],
                            'hit_delta': int(after['hit']) - int(before['hit']),
                            'call_seconds_delta': after['call_seconds'] - before['call_seconds']})
    def group(rows):
        return {'count': len(rows), 'hit_delta_count': sum(r['hit_delta'] for r in rows),
                'hit_pairs': {'both_hit': sum(r['baseline_hit'] and r['quantized_hit'] for r in rows),
                    'lost_hit': sum(r['baseline_hit'] and not r['quantized_hit'] for r in rows),
                    'gained_hit': sum(not r['baseline_hit'] and r['quantized_hit'] for r in rows),
                    'neither_hit': sum(not r['baseline_hit'] and not r['quantized_hit'] for r in rows)},
                'median_call_seconds_delta': statistics.median(r['call_seconds_delta'] for r in rows)}
    return {'overall': group(differences), 'per_case': differences,
            'languages': {k: group([r for r in differences if r['language'] == k])
                          for k in sorted({r['language'] for r in differences})},
            'families': {k: group([r for r in differences if r['family'] == k])
                         for k in sorted({r['family'] for r in differences})},
            'adoption_qualified': False, 'equal_accuracy_qualified': False,
            'native_full_task_qualified': False,
            'scope': 'Dependent authored fixtures only; paired differences are descriptive. '
                     'Separate sessions confound latency. No adoption or equivalence conclusion.'}


def residency(model):
    """Actual full-model tensor storage, with vision subset and quantization-state tensors."""
    seen = set()
    totals = {'all_storage_bytes': 0, 'vision_storage_bytes': 0, 'tensors': []}
    def add(name, tensor):
        if not hasattr(tensor, 'device'):
            return
        if str(tensor.device) != 'cuda:0':
            raise ValueError('Every model/state tensor must reside on cuda:0')
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        size = storage.nbytes() if key not in seen else 0
        seen.add(key)
        vision = 'visual' in name or 'vision' in name
        totals['all_storage_bytes'] += size
        totals['vision_storage_bytes'] += size if vision else 0
        totals['tensors'].append({'name': name, 'device': str(tensor.device),
                                 'dtype': str(tensor.dtype), 'storage_bytes_unique': size,
                                 'vision': vision})
    def state(name, value):
        if hasattr(value, 'device'):
            add(name, value)
        elif isinstance(value, dict):
            for key, item in value.items():
                state(name + '.' + str(key), item)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                state(name + '.' + str(i), item)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        add(name, tensor)
        quant_state = getattr(tensor, 'quant_state', None)
        if quant_state is not None:
            state(name + '.quant_state', quant_state.as_dict(packed=False))
    for name, module in model.named_modules():
        for field in ('CB', 'SCB'):
            add(name + '.state.' + field, getattr(getattr(module, 'state', None), field, None))
    if not totals['vision_storage_bytes']:
        raise ValueError('Actual vision tensor residency required')
    totals['scope'] = 'Unique underlying model/quant-state storage; allocator/device metrics separate'
    return totals
