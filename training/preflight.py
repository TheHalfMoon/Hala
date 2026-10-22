"""Model-independent gates for controlled remote adapter experiments."""

import math
import re
from dataclasses import dataclass

# Optional configuration fields. A configuration that omits them keeps the
# original behaviour, so an existing experiment configuration still validates.
OPTIONAL_FIELDS = frozenset({'supervision', 'coordinate_weight', 'audit_records'})


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    revision: str
    precision: str
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    lora_rank: int
    platform: str
    download_remote_weights: bool
    supervision: str = 'completion'
    coordinate_weight: float = 8.0
    audit_records: int = 0

    @classmethod
    def from_dict(cls, fields):
        if not isinstance(fields, dict) or set(fields) - set(cls.__dataclass_fields__):
            raise ValueError('Exact training configuration fields required')
        required = set(cls.__dataclass_fields__) - set(OPTIONAL_FIELDS)
        if not required <= set(fields):
            raise ValueError('Exact training configuration fields required')
        value = cls(**fields)
        if not isinstance(value.model_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value.model_id):
            raise ValueError('Valid Hugging Face model identifier required')
        if not isinstance(value.revision, str) or not re.fullmatch('[0-9a-f]{40}', value.revision):
            raise ValueError('Immutable model SHA required')
        for name, maximum in (('max_steps', 10000), ('batch_size', 32),
                              ('gradient_accumulation', 128), ('lora_rank', 128)):
            field = getattr(value, name)
            if type(field) is not int or not 1 <= field <= maximum:
                raise ValueError(f'Invalid {name}')
        if (type(value.learning_rate) not in (int, float) or
                not 0 < value.learning_rate <= 0.01 or not math.isfinite(value.learning_rate)):
            raise ValueError('Bounded finite learning rate required')
        if value.precision not in ('fp16', 'bf16'):
            raise ValueError('Only candidate FP16/BF16 training paths are implemented')
        if value.platform not in ('kaggle', 'colab', 'modal', 'linux_cuda'):
            raise ValueError('Remote platform required')
        if type(value.download_remote_weights) is not bool or not value.download_remote_weights:
            raise ValueError('Explicit remote weight-download opt-in required')
        if value.supervision not in ('completion', 'coordinate_weighted'):
            raise ValueError('Only the implemented supervision modes are allowed')
        if (type(value.coordinate_weight) not in (int, float) or
                not 1.0 <= value.coordinate_weight <= 64.0 or
                not math.isfinite(value.coordinate_weight)):
            raise ValueError('Bounded finite coordinate supervision weight required')
        if (type(value.audit_records) is not int or
                not 0 <= value.audit_records <= 1000000):
            raise ValueError('Bounded audit record count required')
        return value


def qualify_environment(config, probe):
    devices = probe.get('devices', [])
    if probe.get('cuda_available') is not True or not devices:
        raise ValueError('Real CUDA runtime required')
    memory = []
    for device in devices:
        capability = device.get('capability', [0, 0])
        vram = device.get('vram_bytes', 0)
        if type(vram) is not int or vram <= 0:
            raise ValueError('Measured per-device memory required')
        if config.precision == 'bf16' and capability[0] < 8:
            raise ValueError('Native BF16 not qualified on this GPU; use FP16')
        memory.append(vram)
    return {'per_device_vram_bytes': min(memory), 'devices': len(devices),
            'memory_pooled': False, 'model_fit_measured': False,
            'precision': config.precision}


def validate_experiment(config, manifest, baseline):
    if (manifest.get('schema') != 'hala.training.manifest.v1' or
            manifest.get('split') != 'train' or manifest.get('rights_cleared') is not True or
            manifest.get('source_kind') not in ('synthetic', 'rights_cleared_public')):
        raise ValueError('Verified training-only source rights required')
    if not re.fullmatch('[0-9a-f]{64}', str(manifest.get('dataset_sha256', ''))):
        raise ValueError('Dataset content digest required')
    families = manifest.get('families')
    held_out = manifest.get('held_out_families')
    if not isinstance(families, list) or not families or not isinstance(held_out, list) or not held_out:
        raise ValueError('Family-level training and held-out manifests required')
    if any(not isinstance(x, str) or not x for x in families + held_out):
        raise ValueError('Invalid family identity')
    if set(families) & set(held_out):
        raise ValueError('Training/evaluation family leakage')
    rate = baseline.get('verified_success_rate')
    if (baseline.get('model_id') != config.model_id or baseline.get('revision') != config.revision or
            baseline.get('evidence') != 'real_inference' or
            type(baseline.get('evaluated_tasks')) is not int or baseline['evaluated_tasks'] <= 0 or
            type(rate) not in (int, float) or not 0 <= rate < 1 or not math.isfinite(rate) or
            not isinstance(baseline.get('deficit'), str) or not baseline['deficit'].strip()):
        raise ValueError('Pinned real unchanged-model baseline with measured deficit required')
    return {'configuration_validated': True, 'artifacts_must_be_hash_verified_at_load': True,
            'license_qualification_required': True, 'training_qualified': False}
