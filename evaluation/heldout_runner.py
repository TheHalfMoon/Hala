"""Remote-only, unchanged Qwen3-VL-2B H14 prospective evaluation entrypoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import time

from evaluation.heldout_fixtures import digest, canonical, verify_freeze, render
from evaluation.heldout_protocol import model_prompt, score, summarize, REVISION
from training.checkpoints import atomic_json

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
MODEL_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
VERSIONS = {'torch': '2.11.0', 'transformers': '5.16.1', 'Pillow': '11.3.0'}


def remote_guard():
    if platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_HELDOUT_ALLOWED') != '1':
        raise ValueError('Approved remote Linux required; local weights forbidden')
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or (
            Path('/kaggle').exists() and not os.environ.get('COLAB_RELEASE_TAG')):
        raise ValueError('Kaggle commercial eligibility remains blocked')


def prepare_images(dataset, font_path, output):
    """Complete and persist the image freeze before any model/processor loading."""
    from PIL import features
    if not features.check('raqm'):
        raise ValueError('Full bilingual run requires actual RAQM')
    provenance = {'font_path': str(font_path), 'font_sha256': digest(font_path),
                  'pillow': importlib.metadata.version('Pillow'), 'raqm': True,
                  'raqm_version': features.version('raqm'), 'images': []}
    image_dir = output / 'images'
    image_dir.mkdir()
    for case in dataset['cases']:
        image = render(case, font_path)
        path = image_dir / (case['id'] + '.png')
        image.save(path, dpi=(case['dpi'], case['dpi']))
        provenance['images'].append({'id': case['id'], 'path': path.name,
                    'size': list(image.size), 'dpi_metadata': case['dpi'],
                    'png_sha256': digest(path),
                    'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        image.close()
    atomic_json(output / 'image-freeze.json', provenance)
    return provenance


def device_memory(torch):
    free, total = torch.cuda.mem_get_info(0)
    return {'allocator_allocated_bytes': torch.cuda.memory_allocated(0),
            'allocator_reserved_bytes': torch.cuda.memory_reserved(0),
            'allocator_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'allocator_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'device_free_bytes': free, 'device_used_bytes': total - free,
            'device_total_bytes': total}


def run(manifest_sha256, output_path, font_path):
    remote_guard()
    dataset = verify_freeze(manifest_sha256)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'hala.h14.result.v1', 'status': 'preparing',
              'manifest_sha256': manifest_sha256, 'protocol': REVISION,
              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
              'training_runs': 0, 'precision': 'fp16', 'attention': 'sdpa',
              'release_qualified': False, 'native_full_task_qualified': False,
              'python': platform.python_version(), 'cases': []}
    def persist():
        atomic_json(output / 'result.json', result)
    def deadline(signum, frame):
        raise TimeoutError('1800-second experiment deadline exceeded')
    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    model = None
    torch = None
    start_all, cpu_start = time.perf_counter(), time.process_time()
    persist()
    try:
        for package, expected in VERSIONS.items():
            actual = importlib.metadata.version(package)
            if actual.split('+')[0] != expected:
                raise ValueError('Pinned remote runtime package mismatch')
        result['packages'] = {p: importlib.metadata.version(p) for p in VERSIONS}
        result['render_provenance'] = prepare_images(dataset, Path(font_path), output)
        result['image_freeze_sha256'] = digest(output / 'image-freeze.json')
        result['all_images_frozen_before_model_load'] = True
        result['status'] = 'images_frozen'
        persist()
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'
        os.environ['HF_HOME'] = '/tmp/hala-model-cache'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch as torch_module
        torch = torch_module
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise ValueError('Actual CUDA allocation required')
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
        start = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
            dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                 trust_remote_code=False)
        torch.cuda.synchronize(0)
        result['download_and_load_seconds'] = time.perf_counter() - start
        result['memory_loaded'] = device_memory(torch)
        result['generation'] = {'do_sample': False, 'num_beams': 1,
                                'max_new_tokens': 48, 'max_time_seconds': 15, 'seed': 42,
                                'deterministic_algorithms': True}
        result['status'] = 'running'
        persist()
        for index, case in enumerate(dataset['cases']):
            image_record = result['render_provenance']['images'][index]
            image_path = output / 'images' / image_record['path']
            if digest(image_path) != image_record['png_sha256']:
                raise ValueError('Frozen input image changed')
            image = Image.open(image_path).convert('RGB')
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
            row = {'id': case['id'], 'family': case['family'], 'language': case['language'],
                   'output': raw, **score(case, raw), 'image_sha256': image_record['png_sha256'],
                   'preprocessing_seconds': preprocessing, 'inference_seconds': inference,
                   'call_seconds': time.perf_counter() - start, 'output_tokens': tokens.shape[1],
                   'call_temperature': 'cold_first_call' if index == 0 else 'warm',
                   'memory': device_memory(torch)}
            result['cases'].append(row)
            persist()
            print(json.dumps({'id': row['id'], 'schema_valid': row['schema_valid'],
                              'hit': row['hit'], 'inference_seconds': inference}), flush=True)
            del inputs, generated, tokens
            image.close()
        result['summary'] = summarize(result['cases'])
        result['memory_final'] = device_memory(torch)
        result['status'] = 'tested'
    except Exception as error:
        result['status'] = 'blocked'
        result['error_type'] = type(error).__name__
        result['evaluated_count'] = len(result['cases'])
        # Partial observations remain inspectable but never get a full-run score.
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
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
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    args = parser.parse_args()
    run(args.manifest_sha256, args.output, args.font)
