"""Bounded directory bundles for one pinned small language LoRA; no ML imports.

The external expected manifest digest and identities are trust anchors supplied by
the caller from the signed archival commit. Self-consistent bytes prove neither
that training happened nor that the adapter improves model quality.
"""
import hashlib
import json
from pathlib import Path
import re
import stat
import struct

BASE = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
LIMITS = {'adapter_model.safetensors': 8 * 1024 * 1024,
          'adapter_config.json': 32 * 1024, 'training_metadata.json': 16 * 1024,
          'manifest.json': 32 * 1024}
TOTAL_LIMIT = 16 * 1024 * 1024
KINDS = {'trained_adapter', 'pretraining_roundtrip'}
TARGET_PATTERN = r'.*language_model.*\.(q_proj|v_proj)$'
# Pinned official text_config: 28 layers, hidden2048, 16 query heads,
# 8 value heads, head_dim128. Rank4 gives 112 tensors /802816 elements.
EXPECTED_SHAPES = {
    f'base_model.model.model.language_model.layers.{layer}.self_attn.{module}.lora_{suffix}.weight': shape
    for layer in range(28)
    for module, output_size in [('q_proj', 2048), ('v_proj', 1024)]
    for suffix, shape in [('A', [4, 2048]), ('B', [output_size, 4])]
}
TENSOR_NAME = re.compile(r'base_model\.model\.model\.language_model\.layers\.([0-9]{1,2})\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight')
CONFIG_FIELDS = {
    'base_model_name_or_path', 'revision', 'peft_type', 'task_type', 'r',
    'target_modules', 'bias', 'lora_alpha', 'lora_dropout', 'auto_mapping',
    'inference_mode', 'modules_to_save', 'init_lora_weights', 'use_rslora',
    'use_dora', 'layers_to_transform', 'layers_pattern', 'rank_pattern',
    'alpha_pattern', 'megatron_config', 'megatron_core', 'loftq_config',
    'layer_replication', 'fan_in_fan_out', 'exclude_modules', 'lora_bias',
    'trainable_token_indices', 'eva_config', 'corda_config', 'qalora_group_size',
    'use_qalora', 'target_parameters', 'ensure_weight_tying',
    'alora_invocation_tokens', 'arrow_config', 'runtime_config',
    'peft_version', 'lora_ga_config', 'monteclora_config', 'use_bdlora', 'velora_config',
}


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, TypeError, RecursionError) as error:
        raise ValueError('Invalid bounded JSON') from error


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(value):
    if not isinstance(value, dict) or set(value) != {
            'base_model', 'base_revision', 'source_commit', 'data_sha256', 'evaluation_sha256'}:
        raise ValueError('Exact immutable identity fields required')
    if value['base_model'] != BASE or value['base_revision'] != REVISION:
        raise ValueError('Only the pinned qualified base is supported')
    for name, length in [('source_commit', 40), ('data_sha256', 64), ('evaluation_sha256', 64)]:
        if not isinstance(value[name], str) or re.fullmatch('[0-9a-f]{' + str(length) + '}', value[name]) is None:
            raise ValueError('Immutable lowercase hash required')


def _redirected(path):
    # lstat exists on Windows Python3.11; pathlib.is_junction does not.
    # Reject every reparse point conservatively, including junctions and symlinks.
    metadata = path.lstat()
    return path.is_symlink() or bool(
        getattr(metadata, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _root(path):
    path = Path(path).absolute()
    if any(_redirected(part) for part in (path, *path.parents)):
        raise ValueError('Symlink or junction directory component forbidden')
    if not path.is_dir():
        raise ValueError('Existing directory required')
    return path


def _read_directory(path, allowed):
    root = _root(path)
    result = {}
    total = 0
    for child in root.iterdir():
        if (child.name not in allowed or _redirected(child)
                or not child.is_file()):
            raise ValueError('Only exact regular allowlisted files accepted')
        size = child.stat().st_size
        if size <= 0 or size > LIMITS[child.name]:
            raise ValueError('File byte budget exceeded or empty file')
        total += size
        if total > TOTAL_LIMIT:
            raise ValueError('Total byte budget exceeded')
        with child.open('rb') as handle:
            raw = handle.read(LIMITS[child.name] + 1)
        if len(raw) != size:
            raise ValueError('Artifact changed during bounded read')
        result[child.name] = raw
    return result


def _config(raw, identity):
    value = _json(raw)
    if not isinstance(value, dict) or set(value) - CONFIG_FIELDS:
        raise ValueError('Unreviewed adapter configuration fields')
    targets = value.get('target_modules')
    valid_targets = targets == TARGET_PATTERN or (
        isinstance(targets, list) and all(isinstance(item, str) for item in targets)
        and sorted(targets) == ['q_proj', 'v_proj'])
    if (value.get('base_model_name_or_path') != identity['base_model'] or
            value.get('revision') not in (None, identity['base_revision']) or
            value.get('peft_type') != 'LORA' or value.get('task_type') != 'CAUSAL_LM' or
            value.get('peft_version') != '0.20.0' or
            type(value.get('r')) is not int or value['r'] != 4 or
            not valid_targets or
            value.get('bias') != 'none' or
            type(value.get('lora_alpha')) is not int or value['lora_alpha'] != 8 or
            type(value.get('lora_dropout')) not in (float, int) or value['lora_dropout'] != 0):
        raise ValueError('Only pinned rank4 language q/v LoRA accepted')
    for key in ('lora_ga_config', 'monteclora_config', 'use_bdlora', 'velora_config'):
        if value.get(key) is not None:
            raise ValueError('Unqualified official PEFT variant')
    if (value.get('init_lora_weights', True) is not True or
            type(value.get('inference_mode', True)) is not bool or
            value.get('megatron_core', 'megatron.core') != 'megatron.core' or
            type(value.get('qalora_group_size', 16)) is not int or
            value.get('qalora_group_size', 16) != 16):
        raise ValueError('Unqualified PEFT initialization or runtime settings')
    # These settings may introduce extra learned weights or alternate code paths.
    for key in ('auto_mapping', 'modules_to_save', 'megatron_config', 'layer_replication',
                'trainable_token_indices', 'target_parameters', 'alora_invocation_tokens',
                'arrow_config', 'runtime_config', 'eva_config', 'corda_config',
                'layers_to_transform', 'layers_pattern', 'exclude_modules'):
        if value.get(key) not in (None, {}, []):
            raise ValueError('Unqualified executable or extra-weight configuration')
    for key in ('use_dora', 'use_rslora', 'use_qalora', 'lora_bias', 'fan_in_fan_out', 'ensure_weight_tying'):
        if value.get(key, False) is not False:
            raise ValueError('Unqualified LoRA variant')
    for key in ('rank_pattern', 'alpha_pattern', 'loftq_config'):
        if value.get(key) not in (None, {}):
            raise ValueError('Unqualified LoRA pattern')
    return value


def _safetensors(raw):
    if len(raw) < 10:
        raise ValueError('Truncated safetensors')
    header_size = struct.unpack('<Q', raw[:8])[0]
    if not 2 <= header_size <= 512 * 1024 or header_size + 8 >= len(raw):
        raise ValueError('Invalid bounded safetensors header length')
    header = _json(raw[8:8 + header_size])
    if not isinstance(header, dict) or not 1 <= len(header) <= 513:
        raise ValueError('Invalid tensor inventory')
    if set(header) - {'__metadata__'} != set(EXPECTED_SHAPES):
        raise ValueError('Complete exact pinned 112-tensor inventory required')
    payload_size = len(raw) - 8 - header_size
    ranges = []
    pairs = {}
    for name, value in header.items():
        if name == '__metadata__':
            if (not isinstance(value, dict) or len(value) > 8 or
                    any(not isinstance(k, str) or not isinstance(v, str) or
                        len(k) > 128 or len(v) > 1024 for k, v in value.items())):
                raise ValueError('Invalid bounded safetensors metadata')
            continue
        match = TENSOR_NAME.fullmatch(name)
        if not match or not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('Only language q/v LoRA tensor names accepted')
        dtype = value['dtype']
        shape = value['shape']
        offsets = value['data_offsets']
        if (dtype not in ('F16', 'BF16', 'F32') or not isinstance(shape, list) or len(shape) != 2 or
                any(type(dim) is not int or not 1 <= dim <= 65536 for dim in shape) or
                shape != EXPECTED_SHAPES[name] or
                shape[0 if match[3] == 'A' else 1] != 4 or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(item) is not int for item in offsets)):
            raise ValueError('Invalid bounded rank4 tensor')
        start, end = offsets
        width = 4 if dtype == 'F32' else 2
        if not 0 <= start < end <= payload_size or end - start != shape[0] * shape[1] * width:
            raise ValueError('Invalid tensor data offsets or length')
        ranges.append((start, end))
        pairs.setdefault((match[1], match[2]), {})[match[3]] = (shape, dtype)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('Tensor overlap, hole or duplicate offset')
        cursor = end
    if cursor != payload_size or not pairs:
        raise ValueError('Unaccounted tensor payload')
    for pair in pairs.values():
        if set(pair) != {'A', 'B'} or pair['A'][1] != pair['B'][1]:
            raise ValueError('Complete matching A/B pair required')
    return len(ranges)


def _validate_files(files, kind, identity):
    required = {'adapter_model.safetensors', 'adapter_config.json'}
    if kind == 'pretraining_roundtrip':
        if files:
            raise ValueError('Pretraining roundtrip contains no trained asset')
        return
    if not required <= set(files):
        raise ValueError('Both adapter safetensors and configuration required')
    _config(files['adapter_config.json'], identity)
    _safetensors(files['adapter_model.safetensors'])
    if 'training_metadata.json' in files:
        value = _json(files['training_metadata.json'])
        if not isinstance(value, dict) or set(value) != {'training_run_id', 'steps', 'seed', 'training_script_sha256'}:
            raise ValueError('Exact bounded training metadata required')
        if (not isinstance(value['training_run_id'], str) or
                re.fullmatch('[A-Za-z0-9_-]{1,80}', value['training_run_id']) is None or
                type(value['steps']) is not int or not 1 <= value['steps'] <= 10000 or
                type(value['seed']) is not int or not 0 <= value['seed'] < 2 ** 32 or
                not isinstance(value['training_script_sha256'], str) or
                re.fullmatch('[0-9a-f]{64}', value['training_script_sha256']) is None):
            raise ValueError('Invalid training metadata values')


def create_bundle(source, destination, identity, *, kind='trained_adapter'):
    """Copy validated exact bytes to a new directory; return external manifest SHA256.

    Caller owns source/destination directories exclusively during this operation.
    Existing destinations are never overwritten. A failed write remains visibly
    incomplete and is not a valid archive until verify_bundle succeeds.
    """
    _identity(identity)
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError('Unknown bundle kind')
    files = _read_directory(source, set(LIMITS) - {'manifest.json'})
    _validate_files(files, kind, identity)
    manifest = {'schema': 'hala.adapter_archive.v1', 'kind': kind,
                'contains_adapter': kind == 'trained_adapter', 'quality_qualified': False,
                'identity': dict(identity), 'files': {
                    name: {'bytes': len(raw), 'sha256': _hash(raw)}
                    for name, raw in sorted(files.items())}}
    raw_manifest = _canonical(manifest)
    destination = Path(destination).absolute()
    _root(destination.parent)
    destination.mkdir(exist_ok=False)
    for name, raw in sorted(files.items()):
        with (destination / name).open('xb') as handle:
            handle.write(raw)
    with (destination / 'manifest.json').open('xb') as handle:
        handle.write(raw_manifest)
    digest = _hash(raw_manifest)
    verify_bundle(destination, identity, digest)
    return digest


def verify_bundle(directory, expected_identity, expected_manifest_sha256):
    """Validate downloaded bytes against externally pinned signed-commit identities."""
    _identity(expected_identity)
    if (not isinstance(expected_manifest_sha256, str) or
            re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256) is None):
        raise ValueError('Externally pinned manifest SHA256 required')
    files = _read_directory(directory, set(LIMITS))
    raw = files.pop('manifest.json', None)
    if raw is None or _hash(raw) != expected_manifest_sha256:
        raise ValueError('Manifest SHA256 mismatch')
    value = _json(raw)
    if (not isinstance(value, dict) or set(value) != {'schema', 'kind', 'contains_adapter',
            'quality_qualified', 'identity', 'files'} or value['schema'] != 'hala.adapter_archive.v1' or
            not isinstance(value['kind'], str) or value['kind'] not in KINDS or value['quality_qualified'] is not False or
            value['contains_adapter'] is not (value['kind'] == 'trained_adapter') or
            value['identity'] != expected_identity or not isinstance(value['files'], dict) or
            set(value['files']) != set(files) or raw != _canonical(value)):
        raise ValueError('Exact canonical manifest and expected identities required')
    for name, data in files.items():
        record = value['files'][name]
        if (not isinstance(record, dict) or set(record) != {'bytes', 'sha256'} or
                type(record['bytes']) is not int or record['bytes'] != len(data) or
                record['sha256'] != _hash(data)):
            raise ValueError('File SHA256 or byte count mismatch')
    _validate_files(files, value['kind'], expected_identity)
    return value
