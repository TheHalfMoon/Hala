import copy
import ast
from contextlib import ExitStack, nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class PrecisionTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('evaluation.development_precision')

    def test_profiles_have_no_offload_and_preserve_reference(self):
        m = self.module()
        from evaluation.development_runtime import SETTINGS
        self.assertEqual(SETTINGS['precision'], 'fp16')
        self.assertEqual(m.profile('fp16')['quantization'], None)
        for mode in ('int8', 'nf4'):
            self.assertFalse(m.profile(mode)['quantization']['llm_int8_enable_fp32_cpu_offload'])
        with self.assertRaises(ValueError):
            m.profile('auto')

    def test_local_rejected_before_freeze_or_ml(self):
        m = self.module()
        with patch('evaluation.development_runtime.platform.system', return_value='Windows'):
            with patch.object(m, 'verify_sources') as verify:
                with self.assertRaises(ValueError):
                    m.run('nf4', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad')
                verify.assert_not_called()

    def test_cache_rejects_aliases_symlinks_and_escape(self):
        m = self.module()
        with patch.dict(os.environ, {'TRANSFORMERS_CACHE': '/elsewhere'}, clear=True):
            with self.assertRaises(ValueError):
                m.check_cache('/tmp/hala-h22-cache')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                m.check_cache('/tmp/hala-h22-cache/../../escape')

    def test_unique_physical_disk_bytes(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a').write_bytes(b'12345')
            os.link(root / 'a', root / 'b')
            self.assertEqual(m.disk_bytes(root), 5)

    def test_cache_measurement_has_entry_and_time_budgets(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(3):
                (root / str(i)).write_bytes(b'x')
            with self.assertRaises(TimeoutError):
                m.disk_bytes(root, max_entries=2)
            with patch.object(m.time, 'perf_counter', side_effect=[0, 10]):
                with self.assertRaises(TimeoutError):
                    m.disk_bytes(root, max_seconds=5)

    def test_cache_admission_budget_exhaustion_rejects_before_ml(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = Path(temp)
            for i in range(3):
                (root / str(i)).write_bytes(b'x')
            with patch.object(m, 'CACHE', temp):
                with self.assertRaises(TimeoutError):
                    m.check_cache(temp, max_entries=2)
                with patch.object(m.time, 'perf_counter', side_effect=[0, 10]):
                    with self.assertRaises(TimeoutError):
                        m.check_cache(temp, max_seconds=5)
            with patch.object(m.runtime, 'evaluation_guard'), patch.object(m, 'verify_sources'), \
                    patch.object(m, 'check_cache', side_effect=TimeoutError('budget')), \
                    patch.object(m.runtime, '_run') as execute, \
                    patch.object(m.importlib.metadata, 'version') as packages:
                with self.assertRaises(TimeoutError):
                    m.run('nf4', 'b' * 64, 'bundle', 'b' * 64, 'out', 'font', 'a' * 40)
                execute.assert_not_called()
                packages.assert_not_called()

    def test_device_samples_follow_logical_cuda_uuid_and_label_missing_fields(self):
        m = self.module()
        uuid = 'GPU-11111111-2222-3333-4444-555555555555'
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(uuid=uuid)))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            actual_uuid = m.resolve_device_uuid(torch)
            with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stdout='10, 20, 30, [Not Supported]')) as query:
                row = m.process_sample(actual_uuid)
                self.assertIn('--id=' + uuid, query.call_args.args[0])
                self.assertNotIn('--id=0', query.call_args.args[0])
                self.assertEqual(row['device_uuid'], uuid)
                self.assertIsNone(row['device']['power_watts'])
                self.assertEqual(row['device_field_errors']['power_watts'], 'unsupported')
            with patch.object(m.subprocess, 'run') as query:
                row = m.process_sample(None)
                query.assert_not_called()
                self.assertIsNone(row['device'])
                self.assertEqual(row['device_error'], 'DeviceIdentityUnavailable')

    def test_sampler_reports_wait_instead_of_fixed_cadence(self):
        m = self.module()
        sampler = m.CostSampler(sample=lambda: {}, limit=1)
        sampler.start()
        result = sampler.stop()
        self.assertEqual(result['minimum_wait_between_samples_seconds'], 2)
        self.assertNotIn('interval_seconds', result)

    def test_sampler_failure_is_bounded_and_preserves_evidence(self):
        m = self.module()
        def broken():
            raise OSError('unavailable')
        sampler = m.CostSampler(sample=broken, limit=2)
        sampler.start()
        result = sampler.stop()
        self.assertLessEqual(len(result['samples']), 2)
        self.assertEqual(result['samples'][0]['error_type'], 'OSError')
        self.assertFalse(result['continuous_peak_qualified'])

    def test_pair_rejects_identity_and_task_changes(self):
        m = self.module()
        from evaluation import development_runtime as runtime
        metadata = {key: 'x' for key in ('source_commit', 'manifest_sha256', 'data_file_sha256',
            'frozen_source', 'model_id', 'model_revision', 'split', 'settings', 'evaluation_provider',
            'runtime_source_sha256', 'packages', 'python', 'cuda', 'gpu', 'gpu_capability',
            'source_freeze_sha256', 'adapter', 'reference_settings')}
        metadata.update(tasks=['a'], mind_tasks=['m'])
        metadata['gpu_uuid'] = 'GPU-11111111-2222-3333-4444-555555555555'
        a = {'status': 'measured', 'precision_profile': m.profile('fp16'),
             'comparison_identity': metadata,
             'episodes': [{'episode': 'a', 'language': 'en', 'family': 'x', 'split': 'dev',
                           'metrics': {'success': False, 'mind_correct': 0}}],
             'mind_cases': [{'id': 'm', 'language': 'ar', 'valid': False, 'correct': False}]}
        a.update({key: value for key, value in metadata.items() if key not in ('tasks', 'mind_tasks')})
        a['precision_profile']['source_freeze_sha256'] = metadata['source_freeze_sha256']
        a['summary'] = runtime.summarize(a['episodes'], a['mind_cases'])
        b = copy.deepcopy(a)
        b['precision_profile'] = m.profile('nf4')
        b['precision_profile']['source_freeze_sha256'] = metadata['source_freeze_sha256']
        self.assertFalse(m.paired(a, b)['adoption_qualified'])
        self.assertTrue(m.paired(a, b)['physical_device_comparison_qualified'])
        changed_device = copy.deepcopy(b)
        changed_device['gpu_uuid'] = 'GPU-66666666-2222-3333-4444-555555555555'
        changed_device['comparison_identity']['gpu_uuid'] = changed_device['gpu_uuid']
        with self.assertRaises(ValueError):
            m.paired(a, changed_device)
        unknown_a, unknown_b = copy.deepcopy(a), copy.deepcopy(b)
        for result in (unknown_a, unknown_b):
            result['gpu_uuid'] = result['comparison_identity']['gpu_uuid'] = None
        self.assertFalse(m.paired(unknown_a, unknown_b)['physical_device_comparison_qualified'])
        for field in ('source_commit', 'tasks', 'frozen_source', 'model_revision', 'adapter', 'settings', 'packages'):
            bad = copy.deepcopy(b)
            bad['comparison_identity'][field] = 'changed'
            with self.assertRaises(ValueError):
                m.paired(a, bad)
        b['episodes'] *= 2
        with self.assertRaises(ValueError):
            m.paired(a, b)

    def test_model_loader_exact_mode_configuration(self):
        m = self.module()
        events = []
        torch = SimpleNamespace(float16='fp16', uint8='uint8')
        model = SimpleNamespace(eval=lambda: model, to=lambda device: model)
        factory = SimpleNamespace(from_pretrained=lambda *a, **kw: (events.append((a, kw)) or model))
        for mode, class_name in (('fp16', ''), ('int8', 'Linear8bitLt'), ('nf4', 'Linear4bit')):
            model.named_modules = lambda: [('layer', type(class_name, (), {})())]
            with patch.dict(sys.modules, {'transformers': SimpleNamespace(BitsAndBytesConfig=lambda **kw: kw)}):
                self.assertIs(m.load_model(torch, factory, m.profile(mode), m.CACHE), model)
            kwargs = events[-1][1]
            self.assertEqual(kwargs['revision'], m.runtime.REVISION)
            self.assertEqual(kwargs['attn_implementation'], 'sdpa')
            if mode != 'fp16':
                self.assertEqual(kwargs['device_map'], {'': 0})
                self.assertFalse(kwargs['quantization_config']['llm_int8_enable_fp32_cpu_offload'])
            else:
                self.assertNotIn('quantization_config', kwargs)
        model.named_modules = lambda: []
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(BitsAndBytesConfig=lambda **kw: kw)}):
            with self.assertRaises(ValueError):
                m.load_model(torch, factory, m.profile('nf4'), m.CACHE)

    def test_reference_ast_and_h19_bytes_unchanged(self):
        m = self.module()
        root = m.ROOT
        revision = '11a3a5359afa896787198e0492ccd036d0b6f4f8'
        old = subprocess.run(['git', 'show', revision + ':evaluation/development_runtime.py'],
                             cwd=root, capture_output=True, check=True).stdout.decode()
        def selected(text):
            return {getattr(n, 'name', 'SETTINGS'): ast.dump(n) for n in ast.parse(text).body
                    if getattr(n, 'name', None) in ('QwenInfer', 'remote_guard') or
                    isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets)}
        self.assertEqual(selected(old), selected((root / 'evaluation/development_runtime.py').read_text()))
        for path in (root / 'development').glob('*.py'):
            frozen = subprocess.run(['git', 'show', revision + ':development/' + path.name],
                                    cwd=root, capture_output=True, check=True).stdout
            self.assertEqual(frozen, path.read_bytes().replace(b'\r\n', b'\n'))

    def test_boolean_settings_profile_rejected(self):
        m = self.module()
        selected = m.profile('nf4')
        selected['reference_settings']['num_beams'] = True
        with self.assertRaises(ValueError):
            m.check_profile(selected)

    def test_freeze_requires_external_digest(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.verify_sources('0' * 64)
        m.verify_sources(m.digest(m.FREEZE))

    def test_precision_runtime_keeps_callback_and_scores_when_sampler_fails(self):
        from tests.test_development_runtime import DevelopmentRuntimeTests
        m = self.module()
        runtime = m.runtime
        for mode in ('fp16', 'int8', 'nf4'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                bundle = root / 'bundle'
                bundle.mkdir()
                (bundle / 'settings.json').write_text(json.dumps(runtime.SETTINGS))
                (bundle / 'dataset.json').write_text(json.dumps({'mind_cases': [],
                    'episodes': [{'id': 'en', 'split': 'dev'}, {'id': 'ar', 'split': 'dev'}]}))
                font = root / 'font'
                font.write_bytes(b'fake')
                seam = DevelopmentRuntimeTests()
                seam.m = runtime
                _, model, processor, _ = seam.seam(root, raw='RAW')
                model.to = lambda device: model
                model.eval = lambda: model
                model.named_modules = lambda: [('layer', type('Linear8bitLt' if mode == 'int8' else 'Linear4bit', (), {})())]
                torch = SimpleNamespace(__version__='2.11.0', float16='fp16', uint8='uint8',
                    manual_seed=lambda _: None, use_deterministic_algorithms=lambda _: None,
                    inference_mode=nullcontext, version=SimpleNamespace(cuda='FAKE'),
                    backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())),
                    cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _: None,
                        get_device_name=lambda _: 'FAKE', get_device_capability=lambda _: (0, 0),
                        reset_peak_memory_stats=lambda _: None, synchronize=lambda _: None, empty_cache=lambda: None))
                def episode(case, infer, *args, **kwargs):
                    self.assertEqual(infer('mind', 'FROZEN PROMPT', None), 'RAW')
                    return {'episode': case['id'], 'language': case['id'], 'family': 'fake', 'split': 'dev',
                            'metrics': {'mind_correct': 0, 'success': False}}
                fake = {'torch': torch, 'accelerate': SimpleNamespace(__version__='1.12.0'),
                    'bitsandbytes': SimpleNamespace(__version__='0.50.2'),
                    'PIL': SimpleNamespace(__version__='11.3.0', Image=SimpleNamespace(open=lambda _: None),
                        features=SimpleNamespace(check_feature=lambda _: True, version_feature=lambda _: '0.10.1')),
                    'transformers': SimpleNamespace(__version__='5.16.1', BitsAndBytesConfig=lambda **kw: kw,
                        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
                        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **kw: model)),
                    'development.freeze': SimpleNamespace(verify_bundle=lambda *a, **kw: {
                        'files': {'dataset.json': 'd' * 64}, 'source': {'fake': True}}),
                    'development.media': SimpleNamespace(FONT_SHA256=hashlib.sha256(font.read_bytes()).hexdigest()),
                    'development.runner': SimpleNamespace(run_episode=episode, run_mind_cases=lambda *a: [])}
                stack.enter_context(patch.dict(sys.modules, fake))
                stack.enter_context(patch.dict(os.environ, {}, clear=True))
                for name, value in [('verify_sources', {}), ('check_cache', None), ('residency', {'fake': True}), ('disk_bytes', 0)]:
                    stack.enter_context(patch.object(m, name, return_value=value))
                if mode == 'int8':
                    stack.enter_context(patch.object(m, 'disk_bytes', side_effect=TimeoutError('budget')))
                stack.enter_context(patch.object(m.CostSampler, 'start', side_effect=RuntimeError('unavailable')))
                stack.enter_context(patch.object(runtime, 'evaluation_guard', return_value={'name': 'fake'}))
                stack.enter_context(patch.object(runtime, 'Path', side_effect=lambda value:
                    SimpleNamespace(resolve=lambda: PurePosixPath(m.CACHE)) if value == m.CACHE else Path(value)))
                stack.enter_context(patch.object(runtime.subprocess, 'run', side_effect=lambda argv, **kw:
                    SimpleNamespace(stdout='a' * 40 if 'rev-parse' in argv else '')))
                stack.enter_context(patch.object(runtime.importlib.metadata, 'version', side_effect=lambda p: dict(runtime.VERSIONS, **m.PINS)[p]))
                stack.enter_context(patch.object(runtime.signal, 'SIGALRM', 14, create=True))
                stack.enter_context(patch.object(runtime.signal, 'alarm', create=True))
                stack.enter_context(patch.object(runtime.signal, 'signal'))
                stack.enter_context(patch.object(runtime, 'memory_sample', return_value={'device_used_bytes': 1}))
                selected = m.profile(mode) | {'source_freeze_sha256': 'b' * 64}
                result = runtime._run(bundle, 'b' * 64, root / 'out', font, 'a' * 40,
                    split='dev', cache_dir=m.CACHE, prepared_adapter=None, precision=selected)
                self.assertEqual(result['status'], 'measured')
                self.assertEqual(result['telemetry_errors']['process_cost_start'], 'RuntimeError')
                self.assertIsNone(result['gpu_uuid'])
                self.assertEqual(result['telemetry_errors']['device_identity'], 'AttributeError')
                if mode == 'int8':
                    self.assertIsNone(result['unique_physical_cache_bytes'])
                    self.assertEqual(result['telemetry_errors']['cache_bytes'], 'TimeoutError')
                self.assertEqual(len(result['calls']), 2)
                self.assertEqual(result['summary']['all']['episodes_complete'], 0)
                self.assertFalse(result['quality_qualified'])
                self.assertEqual(json.loads((root / 'out/result.json').read_text())['precision_profile']['mode'], mode)

    def test_precision_adapter_identity_must_verify_before_execution(self):
        from tests.test_development_adapter import DevelopmentAdapterTests, IDENTITY
        m = self.module()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            archive, digest = DevelopmentAdapterTests().fixture(Path(temp))
            stack.enter_context(patch.object(m.runtime, 'evaluation_guard'))
            stack.enter_context(patch.object(m, 'verify_sources'))
            stack.enter_context(patch.object(m, 'check_cache'))
            stack.enter_context(patch.object(m.importlib.metadata, 'version', side_effect=lambda p: m.PINS[p]))
            execute = stack.enter_context(patch.object(m.runtime, '_run', return_value='candidate'))
            for bad in (dict(IDENTITY, data_sha256='f' * 64), None):
                with self.assertRaises(ValueError):
                    m.run('nf4', 'b' * 64, 'bundle', 'b' * 64, 'out', 'font', 'a' * 40,
                          adapter=archive, adapter_manifest_sha256=digest, adapter_identity=bad)
                execute.assert_not_called()
            self.assertEqual(m.run('nf4', 'b' * 64, 'bundle', 'b' * 64, 'out', 'font', 'a' * 40,
                adapter=archive, adapter_manifest_sha256=digest, adapter_identity=IDENTITY), 'candidate')


if __name__ == '__main__':
    unittest.main()
