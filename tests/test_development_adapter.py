"""CPU boundary evidence; fake tensors are not model qualification."""
from contextlib import ExitStack
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_adapter_archive as archive_tests
from tests import test_development_runtime as runtime_tests
from training.adapter_archive import create_bundle, EXPECTED_SHAPES

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = archive_tests.IDENTITY


class DevelopmentAdapterTests(unittest.TestCase):
    def fixture(self, root):
        source = archive_tests.AdapterArchiveTests().fixture(root)
        archive = root / 'archive'
        digest = create_bundle(source, archive, IDENTITY)
        return archive, digest

    def runtime(self):
        spec = importlib.util.spec_from_file_location('h21d_runtime', ROOT / 'evaluation/development_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_invalid_or_partial_identity_fails_before_model_imports(self):
        runtime = self.runtime()
        original = __import__('builtins').__import__
        imports = []
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'peft', 'safetensors', 'transformers'):
                imports.append(name)
                raise AssertionError('ML import before archive verification')
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            archive, digest = self.fixture(Path(temp))
            cases = [dict(adapter=archive), dict(adapter_manifest_sha256=digest),
                     dict(adapter_identity=IDENTITY),
                     dict(adapter=archive, adapter_manifest_sha256=digest,
                          adapter_identity=dict(IDENTITY, base_revision='main')),
                     dict(adapter=archive, adapter_manifest_sha256='f' * 64, adapter_identity=IDENTITY),
                     dict(adapter=archive, adapter_manifest_sha256=digest,
                          adapter_identity=dict(IDENTITY, extra='bad'))]
            cases += [dict(adapter=archive, adapter_manifest_sha256=digest,
                           adapter_identity=dict(IDENTITY, **{field: value}))
                      for field, value in [('source_commit', 'd' * 40), ('data_sha256', 'd' * 64),
                                           ('evaluation_sha256', 'd' * 64), ('base_model', 'other')]]
            with patch.object(runtime, 'evaluation_guard'), patch('builtins.__import__', side_effect=guarded):
                for kwargs in cases:
                    with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                        runtime.run('unused', 'b' * 64, 'unused', 'unused', 'a' * 40, **kwargs)
            self.assertEqual(imports, [])

    def test_cli_missing_identity_or_unknown_flags_fail_closed(self):
        runtime = self.runtime()
        argv = ['runtime', '--bundle', 'unused', '--manifest-sha256', 'b' * 64,
                '--output', 'unused', '--font-path', 'unused', '--source-commit', 'a' * 40]
        with patch.object(runtime, 'evaluation_guard'), patch.object(runtime, '_run') as execute:
            for flags in (['--adapter', 'unused'], ['--adapter-base-model', runtime.MODEL_ID]):
                with self.subTest(flags=flags), patch.object(sys, 'argv', argv + flags):
                    with self.assertRaises(ValueError):
                        runtime.main()
            with patch.object(sys, 'argv', argv + ['--adapter-url', 'https://invalid.example/']), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runtime.main()
            execute.assert_not_called()

    def test_verified_private_snapshot_preserves_exact_config_and_rejects_missing_or_extra(self):
        from evaluation.development_adapter import verified_adapter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest = self.fixture(root)
            raw = (archive / 'adapter_config.json').read_bytes()
            with verified_adapter(archive, IDENTITY, digest) as prepared:
                snapshot = prepared['directory']
                self.assertNotEqual(snapshot, archive)
                self.assertEqual((snapshot / 'adapter_config.json').read_bytes(), raw)
                (archive / 'adapter_config.json').write_bytes(b'changed original')
                self.assertEqual((snapshot / 'adapter_config.json').read_bytes(), raw)
            self.assertFalse(snapshot.exists())
            (archive / 'adapter_config.json').write_bytes(raw)
            for name in ('extra.pkl', 'adapter_model.safetensors'):
                if name == 'extra.pkl':
                    (archive / name).write_bytes(b'pickle forbidden')
                else:
                    (archive / name).unlink()
                with self.assertRaises(ValueError), verified_adapter(archive, IDENTITY, digest):
                    pass
                if name == 'extra.pkl':
                    (archive / name).unlink()

    def test_roundtrip_kind_is_never_a_candidate(self):
        from evaluation.development_adapter import verified_adapter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'empty'
            source.mkdir()
            archive = root / 'archive'
            digest = create_bundle(source, archive, IDENTITY, kind='pretraining_roundtrip')
            with self.assertRaises(ValueError), verified_adapter(archive, IDENTITY, digest):
                pass

    def tensors(self):
        class Tensor:
            dtype, finite, value = 'float32', True, 1
            def __init__(self, shape):
                self.shape = tuple(shape)
            def detach(self):
                return self
            def cpu(self):
                return self
            def to(self, **kwargs):
                return self
        return {name: Tensor(shape) for name, shape in EXPECTED_SHAPES.items()}

    def test_actual_tensor_inventory_shapes_dtypes_and_every_finite_value_checked(self):
        from evaluation.development_adapter import validate_tensors
        checked = []
        torch = SimpleNamespace(float16='float16', bfloat16='bfloat16', float32='float32',
            isfinite=lambda tensor: (checked.append(tensor) or SimpleNamespace(all=lambda: tensor.finite)))
        tensors = self.tensors()
        validate_tensors(tensors, torch)
        self.assertEqual(len(checked), 112)
        for defect in ('missing', 'extra', 'shape', 'dtype', 'nonfinite'):
            tensors = self.tensors()
            name = next(reversed(tensors))
            if defect == 'missing':
                tensors.pop(name)
            elif defect == 'extra':
                tensors['extra'] = tensors[name]
            elif defect == 'shape':
                tensors[name].shape = (4, 4)
            elif defect == 'dtype':
                tensors[name].dtype = 'int64'
            else:
                tensors[name].finite = False
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_tensors(tensors, torch)

    def test_inference_load_uses_snapshot_exact_bytes_no_training_or_fallback(self):
        from evaluation.development_adapter import verified_adapter, read_tensors, attach_adapter
        order = []
        torch = SimpleNamespace(float16='float16', bfloat16='bfloat16', float32='float32',
            isfinite=lambda tensor: SimpleNamespace(all=lambda: tensor.finite),
            equal=lambda first, second: first.value == second.value)
        loaded = self.tensors()
        model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=False)])
        model.eval = lambda: model
        def load_file(path, **kwargs):
            order.append(('safetensors', Path(path), kwargs))
            return self.tensors()
        def load(base, path, **kwargs):
            order.append(('peft', base, path, kwargs))
            self.assertEqual((path / 'adapter_config.json').read_bytes(), raw)
            return model
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            archive, digest = self.fixture(Path(temp))
            raw = (archive / 'adapter_config.json').read_bytes()
            prepared = stack.enter_context(verified_adapter(archive, IDENTITY, digest))
            stack.enter_context(patch.dict(sys.modules, {
                'safetensors': SimpleNamespace(), 'safetensors.torch': SimpleNamespace(load_file=load_file),
                'peft': SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load),
                                       get_peft_model_state_dict=lambda *a, **kw: loaded)}))
            tensors = read_tensors(prepared, torch)
            self.assertIs(attach_adapter('pinned-base', prepared, torch, tensors), model)
            self.assertEqual([event[0] for event in order], ['safetensors', 'peft'])
            self.assertEqual(order[0][2], {'device': 'cpu'})
            self.assertEqual(order[1][1], 'pinned-base')
            self.assertEqual(order[1][3], {'is_trainable': False, 'local_files_only': True})
            for defect in ('extra', 'value', 'trainable'):
                name = next(iter(loaded))
                if defect == 'extra':
                    loaded['vision.extra'] = loaded[name]
                elif defect == 'value':
                    loaded[name].value = 2
                else:
                    model.parameters = lambda: [SimpleNamespace(requires_grad=True)]
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    attach_adapter('pinned-base', prepared, torch, tensors)
                loaded.pop('vision.extra', None)
                loaded[name].value = 1
            with patch.dict(sys.modules, {'peft': SimpleNamespace(PeftModel=SimpleNamespace(
                    from_pretrained=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('load failed'))),
                    get_peft_model_state_dict=lambda *a, **kw: loaded)}):
                with self.assertRaisesRegex(RuntimeError, 'load failed'):
                    attach_adapter('pinned-base', prepared, torch, tensors)

    def test_baseline_path_and_candidate_classification_are_honest(self):
        runtime = self.runtime()
        with patch.object(runtime, '_run', return_value='baseline') as execute:
            self.assertEqual(runtime.run('bundle', 'b' * 64, 'output', 'font', 'a' * 40), 'baseline')
            self.assertIsNone(execute.call_args.kwargs['prepared_adapter'])
        with tempfile.TemporaryDirectory() as temp:
            archive, digest = self.fixture(Path(temp))
            def execute(*args, prepared_adapter, **kwargs):
                self.assertEqual(prepared_adapter['manifest']['identity'], IDENTITY)
                self.assertEqual(prepared_adapter['manifest_sha256'], digest)
                return 'candidate'
            with patch.object(runtime, 'evaluation_guard'), patch.object(runtime, '_run', side_effect=execute):
                self.assertEqual(runtime.run('bundle', 'b' * 64, 'output', 'font', 'a' * 40,
                    adapter=archive, adapter_manifest_sha256=digest, adapter_identity=IDENTITY), 'candidate')

    def test_full_fake_runtime_load_order_same_tasks_and_honest_saved_results(self):
        """Drive the real runtime control flow without installed ML or model I/O."""
        for provider in ('colab', 'kaggle'):
            with self.subTest(provider=provider):
                self.full_fake_runtime(provider)

    def full_fake_runtime(self, provider):
        from evaluation import development_adapter as adapter_module
        runtime = self.runtime()
        events, tasks = [], []
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            archive, digest = self.fixture(root)
            bundle = root / 'bundle'
            bundle.mkdir()
            (bundle / 'settings.json').write_text(json.dumps(runtime.SETTINGS))
            (bundle / 'dataset.json').write_text(json.dumps({'mind_cases': [{'id': 'mind'}],
                'episodes': [{'id': 'en', 'split': 'dev'}, {'id': 'ar', 'split': 'dev'},
                             {'id': 'heldout', 'split': 'test'}]}))
            font = root / 'font'
            font.write_bytes(b'fake font')
            seam = runtime_tests.DevelopmentRuntimeTests()
            seam.m = runtime
            _, model, processor, _ = seam.seam(root, raw='RAW')
            model.to = lambda _: model
            model.eval = lambda: model
            model.parameters = lambda: [SimpleNamespace(requires_grad=False)]
            torch = SimpleNamespace(__version__='2.11.0', float16='float16', bfloat16='bfloat16',
                float32='float32', manual_seed=lambda _: None, use_deterministic_algorithms=lambda _: None,
                inference_mode=__import__('contextlib').nullcontext,
                version=SimpleNamespace(cuda='fake'),
                isfinite=lambda t: SimpleNamespace(all=lambda: t.finite),
                equal=lambda a, b: a.value == b.value,
                backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())),
                cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _: None,
                    get_device_name=lambda _: 'FAKE', get_device_capability=lambda _: (0, 0),
                    reset_peak_memory_stats=lambda _: None, synchronize=lambda _: None,
                    empty_cache=lambda: None))
            tensors = self.tensors()
            def base_load(*args, **kwargs):
                events.append(('base', args, kwargs))
                return model
            def safe_load(*args, **kwargs):
                events.append(('tensors',))
                return tensors
            def peft_load(*args, **kwargs):
                events.append(('peft',))
                return model
            def mind(cases, infer):
                tasks.extend(case['id'] for case in cases)
                raw = infer('mind', 'FROZEN PROMPT', None)
                return [{'language': 'en', 'valid': True, 'correct': False, 'raw': raw}]
            def episode(case, infer, *args, **kwargs):
                tasks.append(case['id'])
                raw = infer('mind', 'FROZEN PROMPT', None)
                return {'language': case['id'], 'metrics': {'mind_correct': 0, 'success': False}, 'raw': raw}
            modules = {
                'torch': torch,
                'PIL': SimpleNamespace(__version__='11.3.0', Image=SimpleNamespace(open=lambda _: None),
                    features=SimpleNamespace(check_feature=lambda _: True, version_feature=lambda _: '0.10.1')),
                'transformers': SimpleNamespace(__version__='5.16.1',
                    AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
                    Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=base_load)),
                'peft': SimpleNamespace(__version__='0.20.0', PeftModel=SimpleNamespace(from_pretrained=peft_load),
                    get_peft_model_state_dict=lambda *a, **kw: tensors),
                'safetensors': SimpleNamespace(__version__='0.8.0'),
                'safetensors.torch': SimpleNamespace(load_file=safe_load),
                'development.freeze': SimpleNamespace(verify_bundle=lambda *a, **kw: {
                    'files': {'dataset.json': 'd' * 64}, 'source': {'fake': True}}),
                'development.media': SimpleNamespace(FONT_SHA256=hashlib.sha256(font.read_bytes()).hexdigest()),
                'development.runner': SimpleNamespace(run_episode=episode, run_mind_cases=mind)}
            stack.enter_context(patch.dict(sys.modules, modules))
            provenance = ({'name': 'colab', 'colab_release_tag': 'fake-release'} if provider == 'colab'
                          else {'name': 'kaggle', 'kernel_run_type': 'Interactive',
                                'working_directory': '/kaggle/working'})
            admission = stack.enter_context(patch.object(runtime, 'evaluation_guard', return_value=provenance))
            stack.enter_context(patch.object(runtime, 'Path', side_effect=lambda value:
                SimpleNamespace(resolve=lambda: PurePosixPath('/tmp/hala-cache'))
                if value == 'fake-remote-cache' else Path(value)))
            stack.enter_context(patch.object(runtime.subprocess, 'run', side_effect=lambda argv, **kw:
                SimpleNamespace(stdout='a' * 40 if 'rev-parse' in argv else '')))
            pins = dict(runtime.VERSIONS, **adapter_module.VERSIONS)
            stack.enter_context(patch.object(runtime.importlib.metadata, 'version', side_effect=lambda p: pins[p]))
            stack.enter_context(patch.object(runtime.signal, 'SIGALRM', 14, create=True))
            stack.enter_context(patch.object(runtime.signal, 'alarm', create=True))
            stack.enter_context(patch.object(runtime.signal, 'signal'))
            stack.enter_context(patch.object(runtime, 'memory_sample', return_value={'device_used_bytes': 1}))
            original = adapter_module.verify_bundle
            def verified(*args):
                events.append(('archive',))
                return original(*args)
            stack.enter_context(patch.object(adapter_module, 'verify_bundle', side_effect=verified))
            baseline = runtime.run(bundle, 'b' * 64, root / 'baseline', font, 'a' * 40,
                                   cache_dir='fake-remote-cache', provider=provider)
            admission.assert_called_with(provider)
            self.assertEqual(baseline['evaluation_provider'], provenance)
            self.assertEqual(json.loads((root / 'baseline/result.json').read_text())['evaluation_provider'], provenance)
            baseline_tasks = tasks[:]
            self.assertFalse(baseline['adapter_loaded'])
            self.assertEqual(baseline['training_runs'], 0)
            self.assertEqual([e[0] for e in events], ['base'])
            events.clear()
            tasks.clear()
            candidate = runtime.run(bundle, 'b' * 64, root / 'candidate', font, 'a' * 40,
                cache_dir='fake-remote-cache', adapter=archive,
                adapter_manifest_sha256=digest, adapter_identity=IDENTITY, provider=provider)
            admission.assert_called_with(provider)
            self.assertEqual(tasks, baseline_tasks)
            self.assertEqual([c['raw'] for c in candidate['calls']], ['RAW'] * 3)
            self.assertEqual(candidate['summary'], baseline['summary'])
            self.assertEqual([e[0] for e in events], ['archive', 'archive', 'tensors', 'base', 'archive', 'peft'])
            base_args = next(e for e in events if e[0] == 'base')
            self.assertEqual(base_args[1], (runtime.MODEL_ID,))
            self.assertEqual(base_args[2], {'revision': runtime.REVISION, 'trust_remote_code': False,
                'dtype': 'float16', 'attn_implementation': 'sdpa'})
            saved = json.loads((root / 'candidate/result.json').read_text())
            self.assertEqual(saved['evaluation_provider'], provenance)
            self.assertTrue(saved['adapter_loaded'])
            self.assertEqual(saved['adapter']['identity'], IDENTITY)
            self.assertEqual(saved['adapter']['manifest_sha256'], digest)
            self.assertEqual(saved['training_runs'], 0)
            for flag in ('quality_qualified', 'native_qualified', 'execution_authorized'):
                self.assertFalse(saved[flag])


if __name__ == '__main__':
    unittest.main()
