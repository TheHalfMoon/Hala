"""Remote-only frozen shared-base or verified-adapter evaluation; no native actions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
VERSIONS = {'torch': '2.11.0', 'transformers': '5.16.1', 'Pillow': '11.3.0'}
SETTINGS = {'schema': 'hala.development.runtime.v1', 'model_id': MODEL_ID,
            'model_revision': REVISION, 'precision': 'fp16', 'attention': 'sdpa',
            'seed': 42, 'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128,
            'max_input_tokens': 4096,
            'max_time_seconds': 20, 'max_calls_per_episode': 16,
            'max_type_actions_per_episode': 8, 'outer_deadline_seconds': 3600,
            'deterministic_algorithms': True}


def check_digest(value, length=64):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None:
        raise ValueError('External immutable lowercase digest required')


def check_settings(value):
    # JSON canonical comparison distinguishes booleans from integer settings.
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(SETTINGS, sort_keys=True):
        raise ValueError('Frozen settings differ from reviewed unchanged-base runtime')


def remote_guard():
    if (platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG') or
            os.environ.get('HALA_REMOTE_DEVELOPMENT_ALLOWED') != '1' or
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE')):
        raise ValueError('Explicit opt-in and actual Linux Colab marker required; local weights forbidden')


def evaluation_guard(provider='colab'):
    """Admit evaluation only; markers do not qualify GPU availability or packages."""
    if provider == 'colab':
        remote_guard()
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            raise ValueError('Ambiguous evaluation provider markers')
        return {'name': 'colab', 'colab_release_tag': os.environ['COLAB_RELEASE_TAG']}
    if (provider != 'kaggle' or platform.system() != 'Linux' or
            os.environ.get('HALA_REMOTE_DEVELOPMENT_ALLOWED') != '1' or
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE') not in ('Interactive', 'Batch') or
            'COLAB_RELEASE_TAG' in os.environ or not Path('/kaggle/working').is_dir()):
        raise ValueError('Explicit Kaggle evaluation opt-in and actual supported Linux kernel required')
    return {'name': 'kaggle', 'kernel_run_type': os.environ['KAGGLE_KERNEL_RUN_TYPE'],
            'working_directory': '/kaggle/working'}


def persist(path, value):
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False) + '\n').encode('ascii')
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def memory_sample(torch):
    import resource
    free, total = torch.cuda.mem_get_info(0)
    current = None
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            current = int(line.split()[1]) * 1024
    return {'allocated_bytes': torch.cuda.memory_allocated(0),
            'reserved_bytes': torch.cuda.memory_reserved(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'device_used_bytes': total - free, 'device_total_bytes': total,
            'cpu_rss_bytes': current, 'cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def record_memory(record, key, torch, sample=None):
    """Telemetry availability must never decide whether inference evidence exists."""
    try:
        record[key] = (sample or memory_sample)(torch)
    except Exception as error:
        record[key] = None
        record.setdefault('telemetry_errors', {})[key] = type(error).__name__


def finalize_result(result, path, torch, start_all, cpu_start):
    result.update(wall_seconds=time.perf_counter() - start_all,
                  cpu_process_seconds=time.process_time() - cpu_start,
                  model_generation_seconds=sum(r.get('generation_seconds', 0) for r in result['calls']))
    if torch is not None:
        record_memory(result, 'memory_final', torch)
        samples = [result.get(key) for key in ('memory_before_load', 'memory_loaded', 'memory_final')]
        samples += [r.get(key) for r in result['calls'] for key in ('memory_before', 'memory_after')]
        used = [m['device_used_bytes'] for m in samples if isinstance(m, dict) and
                type(m.get('device_used_bytes')) is int and m['device_used_bytes'] >= 0]
        result['max_sampled_device_used_bytes'] = max(used) if used else None
        try:
            torch.cuda.empty_cache()
        except Exception as error:
            result.setdefault('telemetry_errors', {})['empty_cache'] = type(error).__name__
    persist(path, result)


class QwenInfer:
    """Only public prompt/image inputs cross this callback, never task objects."""
    def __init__(self, torch, model, processor, image_open, image_root, sink, sample=memory_sample, deadline_monotonic=None):
        self.torch, self.model, self.processor = torch, model, processor
        self.image_open, self.image_root = image_open, Path(image_root).resolve()
        self.sink, self.sample = sink, sample
        self.deadline_monotonic = deadline_monotonic

    def __call__(self, component, prompt, image_path):
        if self.deadline_monotonic is not None and time.perf_counter() >= self.deadline_monotonic:
            raise TimeoutError('Outer deadline reached; no further model calls permitted')
        if (component not in ('mind', 'type', 'reliance') or not isinstance(prompt, str) or
                not 1 <= len(prompt.encode('utf-8')) <= 16000 or
                (component == 'type') != (image_path is not None)):
            raise ValueError('Bounded public component prompt/image required')
        image, image_hash = None, None
        if image_path is not None:
            path = Path(image_path).resolve()
            if not path.is_relative_to(self.image_root) or not path.is_file():
                raise ValueError('Current rendered frame required')
            if not 8 <= path.stat().st_size <= 2 * 1024 * 1024:
                raise ValueError('Frame byte budget')
            raw_image = path.read_bytes()
            if not raw_image.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('PNG frame required')
            image_hash = hashlib.sha256(raw_image).hexdigest()
            opened = self.image_open(path)
            try:
                if opened.size != (800, 600):
                    raise ValueError('Qualified frame dimensions required')
                image = opened.convert('RGB')
            finally:
                opened.close()
        record = {'component': component, 'prompt': prompt, 'image_sha256': image_hash,
                  'call_index': len(self.sink), 'cold_first_call': not self.sink}
        record_memory(record, 'memory_before', self.torch, self.sample)
        start = time.perf_counter()
        try:
            content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': prompt}]
            template = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                          tokenize=False, add_generation_prompt=True)
            kwargs = {'text': [template], 'return_tensors': 'pt'}
            if image is not None:
                kwargs['images'] = [image]
            inputs = self.processor(**kwargs).to('cuda:0')
            if not 1 <= inputs.input_ids.shape[1] <= SETTINGS['max_input_tokens']:
                raise ValueError('Expanded model input token budget exceeded')
            self.torch.cuda.synchronize(0)
            generate_start = time.perf_counter()
            record['preprocessing_seconds'] = generate_start - start
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                                max_new_tokens=128, max_time=20)
            self.torch.cuda.synchronize(0)
            record['generation_seconds'] = time.perf_counter() - generate_start
            completion = generated[:, inputs.input_ids.shape[1]:]
            record['output_tokens'] = completion.shape[1]
            raw = self.processor.batch_decode(completion, skip_special_tokens=True)[0]
            if not isinstance(raw, str):
                raise ValueError('Raw decoded string required')
            record['raw'] = raw
            return raw
        except Exception as error:
            record['error_type'] = type(error).__name__
            raise
        finally:
            if image is not None:
                image.close()
            record['wall_seconds'] = time.perf_counter() - start
            record_memory(record, 'memory_after', self.torch, self.sample)
            self.sink.append(record)


def summarize(episodes, mind):
    result = {}
    for language in ('en', 'ar', 'all'):
        selected = [r for r in episodes if language == 'all' or r['language'] == language]
        decisions = [r for r in mind if language == 'all' or r['language'] == language]
        if any(type(r['metrics'].get('mind_correct')) is not int for r in selected):
            raise ValueError('Reviewed independent episode Mind semantic scoring required')
        result[language] = {'episodes': len(selected),
            'episodes_complete': sum(bool(r['metrics']['success']) for r in selected),
            'mind_cases': len(decisions), 'mind_case_exact_correct': sum(bool(r['correct']) for r in decisions),
            'mind_invalid': sum(not r['valid'] for r in decisions)}
        for key in ('model_calls', 'model_errors', 'unsafe_proposals', 'unnecessary_calls',
                    'recoveries', 'action_errors', 'environment_errors', 'action_calls',
                    'mind_calls', 'mind_correct', 'mind_errors', 'plan_calls', 'plan_correct'):
            result[language][key] = sum(r['metrics'].get(key, 0) for r in selected)
        result[language]['mind_semantic_accuracy'] = (result[language]['mind_correct'] / result[language]['mind_calls']
                                                    if result[language]['mind_calls'] else None)
    return result


def run(bundle, manifest_sha256, output, font_path, source_commit, *, split='dev', cache_dir='/tmp/hala-h21-cache',
        adapter=None, adapter_manifest_sha256=None, adapter_identity=None, provider='colab'):
    supplied = (adapter, adapter_manifest_sha256, adapter_identity)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise ValueError('Adapter directory, manifest SHA256 and full external identity required together')
        evaluation_guard(provider)
        from evaluation.development_adapter import verified_adapter
        with verified_adapter(adapter, adapter_identity, adapter_manifest_sha256) as prepared:
            return _run(bundle, manifest_sha256, output, font_path, source_commit,
                        split=split, cache_dir=cache_dir, prepared_adapter=prepared, provider=provider)
    return _run(bundle, manifest_sha256, output, font_path, source_commit,
                split=split, cache_dir=cache_dir, prepared_adapter=None, provider=provider)


def _run(bundle, manifest_sha256, output, font_path, source_commit, *, split, cache_dir, prepared_adapter, provider='colab'):
    admitted_provider = evaluation_guard(provider)
    check_digest(manifest_sha256)
    check_digest(source_commit, 40)
    if split not in ('dev', 'test'):
        raise ValueError('Only independent dev or test episodes allowed')
    from development.freeze import verify_bundle
    from development.media import FONT_SHA256
    from development.runner import run_episode, run_mind_cases
    bundle = Path(bundle)
    manifest = verify_bundle(bundle, require_media=True, expected_manifest_sha256=manifest_sha256)
    check_settings(json.loads((bundle / 'settings.json').read_text(encoding='utf-8')))
    if hashlib.sha256(Path(font_path).read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('Exact qualified font required before model allocation')
    root = Path(__file__).resolve().parents[1]
    actual_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all', '--', 'development',
                            'evaluation/development_runtime.py', 'evaluation/development_adapter.py',
                            'training/adapter_archive.py'], cwd=root, capture_output=True, text=True, check=True).stdout
    if actual_commit != source_commit or dirty:
        raise ValueError('Exact published clean source commit required')
    cache = Path(cache_dir).resolve()
    if not (str(cache).startswith('/tmp/hala-') or cache.is_relative_to('/content')):
        raise ValueError('Remote approved cache path required')
    pins = dict(VERSIONS)
    if prepared_adapter is not None:
        from evaluation.development_adapter import VERSIONS as adapter_versions
        pins.update(adapter_versions)
    packages = {p: importlib.metadata.version(p) for p in pins}
    if any(packages[p].split('+')[0] != expected for p, expected in pins.items()):
        raise ValueError('Exact runtime package pins required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames = output / 'frames'
    frames.mkdir()
    result = {'schema': 'hala.development.baseline.v1', 'status': 'preparing',
              'evaluation_provider': admitted_provider,
              'model_id': MODEL_ID, 'model_revision': REVISION, 'settings': SETTINGS,
              'source_commit': source_commit, 'runtime_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'manifest_sha256': manifest_sha256, 'data_file_sha256': manifest['files']['dataset.json'],
              'frozen_source': manifest['source'], 'packages': packages, 'python': platform.python_version(),
              'split': split, 'calls': [], 'episodes': [], 'mind_cases': [], 'training_runs': 0,
              'adapter_loaded': False, 'quality_qualified': False, 'native_qualified': False,
              'execution_authorized': False, 'memory_sampling': 'Before/after load and calls; full-device samples are not a continuous peak.'}
    if prepared_adapter is not None:
        result.update(schema='hala.development.adapter_evaluation.v1', adapter={
            'manifest_sha256': prepared_adapter['manifest_sha256'],
            'identity': prepared_adapter['manifest']['identity'],
            'files': prepared_adapter['manifest']['files'], 'kind': 'trained_adapter'})
    result_path = output / 'result.json'
    persist(result_path, result)
    start_all, cpu_start = time.perf_counter(), time.process_time()
    deadline_monotonic = start_all + SETTINGS['outer_deadline_seconds']
    torch = model = None
    def deadline(_signum, _frame):
        raise TimeoutError('Bounded outer experiment deadline exceeded')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SETTINGS['outer_deadline_seconds'])
    try:
        os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                          HF_HOME=str(cache), HF_HUB_CACHE=str(cache),
                          CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import torch as torch_module
        from PIL import Image, features, __version__ as pillow_version
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch = torch_module
        actual_packages = {'torch': torch.__version__, 'transformers': transformers.__version__, 'Pillow': pillow_version}
        if any(actual_packages[p].split('+')[0] != expected for p, expected in VERSIONS.items()):
            raise ValueError('Imported runtime differs from exact package pins')
        result['imported_packages'] = actual_packages
        if not torch.cuda.is_available() or not features.check_feature('raqm') or features.version_feature('raqm') != '0.10.1':
            raise ValueError('Actual CUDA and qualified RAQM required')
        if prepared_adapter is not None:
            import peft
            import safetensors
            from evaluation.development_adapter import read_tensors
            actual_adapter_packages = {'peft': peft.__version__, 'safetensors': safetensors.__version__}
            if any(actual_adapter_packages[p] != expected for p, expected in adapter_versions.items()):
                raise ValueError('Imported adapter packages differ from exact pins')
            result['imported_packages'].update(actual_adapter_packages)
            adapter_tensors = read_tensors(prepared_adapter, torch)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      gpu_capability=list(torch.cuda.get_device_capability(0)))
        record_memory(result, 'memory_before_load', torch)
        torch.cuda.reset_peak_memory_stats(0)
        start_load = time.perf_counter()
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION,
                    trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
        if prepared_adapter is not None:
            from evaluation.development_adapter import attach_adapter
            model = attach_adapter(model, prepared_adapter, torch, adapter_tensors)
            del adapter_tensors
            result['adapter_loaded'] = True
        torch.cuda.synchronize(0)
        result.update(load_seconds=time.perf_counter() - start_load, status='running')
        record_memory(result, 'memory_loaded', torch)
        persist(result_path, result)
        infer = QwenInfer(torch, model, processor, Image.open, frames, result['calls'], deadline_monotonic=deadline_monotonic)
        dataset = json.loads((bundle / 'dataset.json').read_text(encoding='utf-8'))
        # Standalone Mind diagnostics remain the explicitly labelled test split,
        # even if the closed-loop episode allocation chosen here is dev.
        for case in dataset['mind_cases']:
            if time.perf_counter() >= deadline_monotonic:
                raise TimeoutError('Outer experiment deadline reached')
            result['mind_cases'].extend(run_mind_cases([case], infer))
            persist(result_path, result)
        for episode in (e for e in dataset['episodes'] if e['split'] == split):
            if time.perf_counter() >= deadline_monotonic:
                raise TimeoutError('Outer experiment deadline reached')
            result['episodes'].append(run_episode(episode, infer, frames, font_path, max_calls=16))
            persist(result_path, result)
        result.update(summary=summarize(result['episodes'], result['mind_cases']), status='measured')
    except Exception as error:
        result.update(status='blocked', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        finalize_result(result, result_path, torch, start_all, cpu_start)
        if model is not None:
            del model
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font-path', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--cache-dir', default='/tmp/hala-h21-cache')
    parser.add_argument('--provider', choices=['colab', 'kaggle'], default='colab',
                        help='Evaluation provider; Kaggle requires explicit selection and real runtime markers')
    parser.add_argument('--adapter', help='Operator-downloaded local archive directory on the approved remote runtime')
    parser.add_argument('--adapter-manifest-sha256')
    for field in ('base-model', 'base-revision', 'source-commit', 'data-sha256', 'evaluation-sha256'):
        parser.add_argument('--adapter-' + field)
    args = parser.parse_args()
    identity = {field: getattr(args, 'adapter_' + field) for field in
                ('base_model', 'base_revision', 'source_commit', 'data_sha256', 'evaluation_sha256')}
    if all(value is None for value in identity.values()):
        identity = None
    result = run(args.bundle, args.manifest_sha256, args.output, args.font_path, args.source_commit,
                 split=args.split, cache_dir=args.cache_dir, adapter=args.adapter,
                 adapter_manifest_sha256=args.adapter_manifest_sha256, adapter_identity=identity, provider=args.provider)
    print(json.dumps({'status': result['status'], 'summary': result['summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
