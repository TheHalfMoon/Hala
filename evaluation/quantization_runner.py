"""Opt-in Colab T4 H17 experiment. Run one prespecified mode per fresh session."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import time

from evaluation.heldout_fixtures import digest
from evaluation.heldout_protocol import model_prompt, score, summarize, REVISION
from evaluation.heldout_runner import device_memory, MODEL_ID, MODEL_REVISION
from evaluation.quantization import (CACHE, FREEZE, remote_guard, verify_sources,
                                     verify_media, quantization_config, paired, residency)
from training.checkpoints import atomic_json


def run(mode, image_dir, output_path, source_freeze_sha256):
    remote_guard()  # Before importing torch, transformers, bitsandbytes or processor.
    settings = quantization_config(mode)
    config, dataset, baseline = verify_sources(source_freeze_sha256)
    paths = verify_media(baseline, Path(image_dir))
    output = Path(output_path)
    if not output.resolve().is_relative_to(Path('/tmp')) or output.is_symlink():
        raise ValueError('Ephemeral /tmp output required')
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'hala.h17.quantization.v1', 'status': 'preparing', 'mode': mode,
              'cases': [], 'source_freeze_sha256': digest(FREEZE),
              'manifest_sha256': baseline['manifest_sha256'], 'protocol': REVISION,
              'baseline_sha256': config['baseline_sha256'],
              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
              'processor_revision': MODEL_REVISION, 'attention': 'sdpa',
              'quantization_config': settings, 'generation': baseline['generation'],
              'training_runs': 0, 'adoption_qualified': False, 'release_qualified': False,
              'equal_accuracy_qualified': False, 'native_full_task_qualified': False,
              'python': platform.python_version(), 'cost_scope': {
                  'paid_actions': 0, 'scope': 'Free opted-in Colab session only. '
                  'Wall/device/allocator measures, no commercial serving cost estimate.'}}
    def persist():
        atomic_json(output / 'result.json', result)
    def deadline(signum, frame):
        raise TimeoutError('1800-second experiment deadline exceeded')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    start_all, cpu_start = time.perf_counter(), time.process_time()
    torch = model = None
    persist()
    try:
        result['packages'] = {p: importlib.metadata.version(p) for p in config['packages']}
        if result['packages'] != config['packages']:
            raise ValueError('Exact pinned versions including CUDA wheel required')
        for key in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
                    'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME'):
            os.environ[key] = CACHE
        os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import torch as torch_module
        torch = torch_module
        from PIL import Image
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if (not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0)
                or torch.version.cuda != '13.0' or torch.cuda.get_device_capability(0) != (7, 5)):
            raise ValueError('Actual pinned CUDA13 Tesla T4 required')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result['gpu'] = torch.cuda.get_device_name(0)
        result['cuda'] = torch.version.cuda
        result['memory_before_load'] = device_memory(torch)
        torch.cuda.reset_peak_memory_stats(0)
        kwargs = dict(settings)
        if mode == 'nf4':
            kwargs['bnb_4bit_compute_dtype'] = torch.float16
            kwargs['bnb_4bit_quant_storage'] = torch.uint8
        start = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
            dtype=torch.float16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(**kwargs), device_map={'': 0},
            cache_dir=CACHE).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                 trust_remote_code=False, cache_dir=CACHE)
        torch.cuda.synchronize(0)
        result['download_and_load_seconds'] = time.perf_counter() - start
        result['memory_loaded'] = device_memory(torch)
        result['residency_loaded'] = residency(model)
        result['quantized_module_names'] = [name for name, module in model.named_modules()
                                           if type(module).__name__ ==
                                           ('Linear8bitLt' if mode == 'int8' else 'Linear4bit')]
        if not result['quantized_module_names']:
            raise ValueError('Actual quantized linear modules required')
        result['status'] = 'running'
        persist()
        for index, (case, path) in enumerate(zip(dataset['cases'], paths)):
            if digest(path) != baseline['cases'][index]['image_sha256']:
                raise ValueError('Frozen media changed during run')
            with Image.open(path) as source:
                image = source.convert('RGB')
            media = baseline['render_provenance']['images'][index]
            import hashlib
            if (list(image.size) != case['size'] or
                    hashlib.sha256(image.tobytes()).hexdigest() != media['rgb_sha256']):
                raise ValueError('Frozen RGB/dimension mismatch')
            messages = [{'role': 'user', 'content': [{'type': 'image'},
                        {'type': 'text', 'text': model_prompt(case)}]}]
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            template = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            inputs = processor(text=[template], images=[image], return_tensors='pt').to('cuda:0')
            torch.cuda.synchronize(0)
            preprocessing = time.perf_counter() - start
            generate_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                           num_beams=1, max_time=15)
            torch.cuda.synchronize(0)
            inference = time.perf_counter() - generate_start
            tokens = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            result['cases'].append({'id': case['id'], 'language': case['language'],
                'family': case['family'], 'output': raw, **score(case, raw),
                'image_sha256': media['png_sha256'], 'preprocessing_seconds': preprocessing,
                'inference_seconds': inference, 'call_seconds': time.perf_counter() - start,
                'output_tokens': tokens.shape[1], 'memory': device_memory(torch),
                'call_temperature': 'cold_first_call' if index == 0 else 'warm'})
            persist()
            del inputs, generated, tokens
            image.close()
        result['residency_final'] = residency(model)
        result['memory_final'] = device_memory(torch)
        result['summary'] = summarize(result['cases'])
        result['paired'] = paired(baseline['cases'], result['cases'])
        result['status'] = 'tested_experimental'
    except Exception as error:
        result['status'] = 'blocked'
        result['error_type'] = type(error).__name__
        result['error'] = str(error)
        result['evaluated_count'] = len(result['cases'])
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        result['wall_seconds'] = time.perf_counter() - start_all
        result['process_cpu_seconds'] = time.process_time() - cpu_start
        persist()
        if model is not None:
            del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('int8', 'nf4'), required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    args = parser.parse_args()
    run(args.mode, args.images, args.output, args.source_freeze_sha256)
