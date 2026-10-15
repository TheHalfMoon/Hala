"""Explicit small trained-adapter inference; archive checks need no ML imports."""
from contextlib import contextmanager
from pathlib import Path
import tempfile

from training.adapter_archive import EXPECTED_SHAPES, LIMITS, verify_bundle

VERSIONS = {'peft': '0.20.0', 'safetensors': '0.8.0'}


@contextmanager
def verified_adapter(directory, identity, manifest_sha256):
    """Snapshot exact bounded bytes privately before any model imports/allocation.

    Both source and snapshot must match externally pinned archive identities.
    Configuration is copied verbatim, including a producer's null revision.
    """
    manifest = verify_bundle(directory, identity, manifest_sha256)
    if manifest['kind'] != 'trained_adapter':
        raise ValueError('A trained adapter archive is required for candidate evaluation')
    with tempfile.TemporaryDirectory(prefix='hala-verified-adapter-') as temp:
        snapshot = Path(temp)
        for name in ('manifest.json', *manifest['files']):
            with (Path(directory) / name).open('rb') as stream:
                raw = stream.read(LIMITS[name] + 1)
            if len(raw) > LIMITS[name]:
                raise ValueError('Archive changed beyond bounded size')
            (snapshot / name).write_bytes(raw)
        verify_bundle(snapshot, identity, manifest_sha256)
        yield {'directory': snapshot, 'manifest': manifest, 'manifest_sha256': manifest_sha256}


def validate_tensors(tensors, torch):
    if set(tensors) != set(EXPECTED_SHAPES):
        raise ValueError('Actual safetensors must contain all exact 112 adapter tensors')
    for name, tensor in tensors.items():
        if (tuple(tensor.shape) != tuple(EXPECTED_SHAPES[name]) or
                tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
                not torch.isfinite(tensor).all()):
            raise ValueError('Actual adapter tensor shape, dtype or finite check failed')


def read_tensors(prepared, torch):
    """Actual safe-format CPU decode after remote/package gates, before base load."""
    from safetensors.torch import load_file
    tensors = load_file(str(prepared['directory'] / 'adapter_model.safetensors'), device='cpu')
    validate_tensors(tensors, torch)
    return tensors


def attach_adapter(base, prepared, torch, expected_tensors):
    from peft import PeftModel, get_peft_model_state_dict
    # Recheck the private snapshot before PEFT's own safe-format loading.
    verify_bundle(prepared['directory'], prepared['manifest']['identity'], prepared['manifest_sha256'])
    model = PeftModel.from_pretrained(base, prepared['directory'],
                                     is_trainable=False, local_files_only=True).eval()
    # PEFT can warn rather than fail on missing weights. Inspect actual loaded
    # weights to reject additional initialized modules, dropped keys or values.
    loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    validate_tensors(loaded, torch)
    for name, tensor in loaded.items():
        if not torch.equal(tensor.detach().cpu(), expected_tensors[name].to(dtype=tensor.dtype)):
            raise ValueError('Loaded adapter weights differ from verified safetensors')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Inference adapter and base must both be frozen')
    return model
