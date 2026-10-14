"""Immutable checkpoint versions; one atomic pointer publishes a complete pair."""

import json
import math
import os
from pathlib import Path
import re
import uuid

from training.artifacts import artifact_path, sha256_file, strict_json

IDENTITY_FIELDS = ('config_sha256', 'dataset_sha256', 'manifest_sha256')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    # Linux training uses directory fsync for durable rename publication.
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_state(state):
    required = {*IDENTITY_FIELDS, 'step', 'max_steps', 'metrics',
                'changed_parameter_tensors', 'peak_vram_bytes', 'runtime'}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError('Exact checkpoint state required')
    if any(not isinstance(state[name], str) or not re.fullmatch('[0-9a-f]{64}', state[name])
           for name in IDENTITY_FIELDS):
        raise ValueError('Checkpoint input identities required')
    if (type(state['step']) is not int or type(state['max_steps']) is not int or
            not 1 <= state['step'] <= state['max_steps'] <= 10000 or
            type(state['changed_parameter_tensors']) is not int or
            not 0 <= state['changed_parameter_tensors'] <= 100000 or
            type(state['peak_vram_bytes']) is not int or state['peak_vram_bytes'] < 0 or
            not isinstance(state['runtime'], dict) or
            not isinstance(state['metrics'], list) or len(state['metrics']) != state['step']):
        raise ValueError('Bounded consistent checkpoint progress required')
    for index, row in enumerate(state['metrics'], start=1):
        if (not isinstance(row, dict) or set(row) != {'step', 'loss', 'seconds'} or
                type(row['step']) is not int or row['step'] != index or
                any(type(row[name]) not in (int, float) or not math.isfinite(row[name]) or
                    row[name] < 0 for name in ('loss', 'seconds'))):
            raise ValueError('Complete ordered finite checkpoint metrics required')
    # Strict encoding also bounds the supported torch-safe metadata representation.
    if len(json.dumps(state, allow_nan=False).encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('Checkpoint metadata budget exceeded')


def file_hashes(version):
    hashes = {}
    for path in sorted(version.rglob('*')):
        if path.is_symlink():
            raise ValueError('Checkpoint symlink forbidden')
        if path.is_file() and path != version / 'checkpoint.json':
            relative = path.relative_to(version).as_posix()
            artifact_path(version, relative)
            hashes[relative] = sha256_file(path)
    if (not hashes or 'resume.pt' not in hashes or
            not any(name.startswith('adapter/') and name.endswith('.safetensors') for name in hashes) or
            len(hashes) > 1024):
        raise ValueError('Complete adapter and optimizer artifacts required')
    return hashes


def validate_version(version, expected=None):
    metadata_path = version / 'checkpoint.json'
    if not metadata_path.is_file() or metadata_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Bounded complete checkpoint metadata required')
    metadata = strict_json(metadata_path.read_text(encoding='utf-8'))
    if (not isinstance(metadata, dict) or set(metadata) != {'schema', 'state', 'artifacts'} or
            metadata['schema'] != 'hala.training.checkpoint.v1'):
        raise ValueError('Unknown checkpoint schema')
    state = metadata['state']
    validate_state(state)
    if expected is not None and any(state[name] != expected.get(name)
                                    for name in (*IDENTITY_FIELDS, 'max_steps')):
        raise ValueError('Resume configuration/data/manifest identity mismatch')
    if metadata['artifacts'] != file_hashes(version):
        raise ValueError('Checkpoint artifact digest or file-set mismatch')
    return state, metadata['artifacts']


def load_checkpoint(output, expected):
    output = Path(output)
    pointer_path = output / 'latest.json'
    if not pointer_path.is_file() or pointer_path.stat().st_size > 1024:
        raise ValueError('Complete latest checkpoint pointer required')
    pointer = strict_json(pointer_path.read_text(encoding='utf-8'))
    if (not isinstance(pointer, dict) or set(pointer) != {'schema', 'version', 'checkpoint_sha256'} or
            pointer['schema'] != 'hala.training.latest.v1' or
            not isinstance(pointer['version'], str) or
            not re.fullmatch(r'checkpoints/step-[0-9]{8}-[0-9a-f]{32}', pointer['version']) or
            not isinstance(pointer['checkpoint_sha256'], str) or
            not re.fullmatch('[0-9a-f]{64}', pointer['checkpoint_sha256'])):
        raise ValueError('Invalid checkpoint pointer')
    version = artifact_path(output.resolve(), pointer['version'])
    if sha256_file(version / 'checkpoint.json') != pointer['checkpoint_sha256']:
        raise ValueError('Checkpoint metadata digest mismatch')
    state, _ = validate_version(version, expected)
    return version, state


def publish_checkpoint(output, state, writer):
    """writer receives a staging directory. Old published artifacts never change."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    validate_state(state)
    if (output / 'latest.json').exists():
        _, previous = load_checkpoint(output, state)
        if previous['step'] >= state['step']:
            raise ValueError('Refusing to overwrite published checkpoint progress')
    versions = output / 'checkpoints'
    versions.mkdir(exist_ok=True)
    identity = uuid.uuid4().hex
    stage = versions / ('.staging-' + identity)
    version = versions / f"step-{state['step']:08d}-{identity}"
    stage.mkdir()
    # An interrupted staging directory is harmless: resume consults only latest.json.
    writer(stage)
    hashes = file_hashes(stage)
    atomic_json(stage / 'checkpoint.json', {'schema': 'hala.training.checkpoint.v1',
                                           'state': state, 'artifacts': hashes})
    validate_version(stage, state)
    for path in stage.rglob('*'):
        if path.is_file():
            with path.open('r+b') as handle:
                os.fsync(handle.fileno())
    for path in sorted((p for p in stage.rglob('*') if p.is_dir()), reverse=True):
        sync_directory(path)
    sync_directory(stage)
    os.rename(stage, version)
    sync_directory(versions)
    atomic_json(output / 'latest.json', {'schema': 'hala.training.latest.v1',
                                        'version': version.relative_to(output).as_posix(),
                                        'checkpoint_sha256': sha256_file(version / 'checkpoint.json')})
    return version


def completed_checkpoint_result(version, state, config_data):
    validate_state(state)
    if state['step'] != state['max_steps'] or not state['changed_parameter_tensors']:
        raise ValueError('Completed checkpoint with verifiable adapter update required')
    _, artifacts = validate_version(Path(version), state)
    return {'schema': 'hala.training.run.v1', 'configuration': config_data,
            **{name: state[name] for name in IDENTITY_FIELDS},
            'runtime': state['runtime'], 'metrics': state['metrics'],
            'changed_parameter_tensors': state['changed_parameter_tensors'],
            'peak_vram_bytes': state['peak_vram_bytes'], 'artifacts': artifacts,
            'checkpoint': Path(version).name, 'evaluation_required': True,
            'adapter_qualified': False}
