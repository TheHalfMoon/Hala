"""Bounded remote GPU qualification; no model download or checkpoint training."""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'


def probe():
    result = {'schema': 'hala.runtime.probe.v1', 'python': platform.python_version(),
              'platform': platform.platform(), 'timestamp_unix': time.time(),
              'downloads_weights': False, 'trains_checkpoint': False, 'devices': [],
              'packages': {}}
    for package in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    try:
        import torch
    except ImportError:
        result.update(status='blocked', reason='torch_not_installed')
        return result
    result['cuda_build'] = torch.version.cuda
    result['cuda_available'] = torch.cuda.is_available()
    if not result['cuda_available']:
        result.update(status='blocked', reason='cuda_unavailable')
        return result
    torch.manual_seed(42)
    for index in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{index}')
        properties = torch.cuda.get_device_properties(index)
        row = {'index': index, 'name': properties.name,
               'vram_bytes': properties.total_memory,
               'capability': list(torch.cuda.get_device_capability(index)),
               'bf16_native_candidate': properties.major >= 8}
        torch.cuda.reset_peak_memory_stats(index)
        x = torch.randn(64, 64, device=device, dtype=torch.float16)
        weight = torch.randn(64, 64, device=device, dtype=torch.float16, requires_grad=True)
        before = weight.detach().clone()
        optimizer = torch.optim.SGD([weight], lr=0.01)
        torch.cuda.synchronize(index)
        start = time.perf_counter()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = ((x @ weight).float().square()).mean()
            loss.backward()
            if not torch.isfinite(weight.grad).all():
                raise RuntimeError('Nonfinite FP16 gradients')
            optimizer.step()
        torch.cuda.synchronize(index)
        row['four_fp16_steps_seconds'] = time.perf_counter() - start
        row['tensor_gradient_changed'] = bool(not torch.equal(before, weight.detach()))
        row['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(index)
        row['final_loss'] = loss.item()
        row['qualification_scope'] = 'small_tensor_fp16_backward_only'
        result['devices'].append(row)
        del x, weight, before, optimizer
        torch.cuda.empty_cache()
    result['status'] = 'tested'
    result['model_runtime_qualified'] = False
    return result


if __name__ == '__main__':
    output = probe()
    encoded = json.dumps(output, indent=2, allow_nan=False).encode()
    destination = Path('/kaggle/working/runtime-probe.json') if Path('/kaggle/working').exists() else Path('runtime-probe.json')
    destination.write_bytes(encoded)
    destination.with_suffix('.sha256').write_text(hashlib.sha256(encoded).hexdigest(), encoding='ascii')
    print(encoded.decode())
