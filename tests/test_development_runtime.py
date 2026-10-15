import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('h21_runtime', Path(__file__).resolve().parents[1] / 'evaluation/development_runtime.py')


class DevelopmentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(cls.m)

    def test_guard_before_any_model_import(self):
        with patch.object(self.m.platform, 'system', return_value='Windows'), patch.dict(self.m.os.environ, {'COLAB_RELEASE_TAG': 'real', 'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1'}, clear=True):
            with self.assertRaises(ValueError):
                self.m.remote_guard()
        with patch.object(self.m.platform, 'system', return_value='Linux'), patch.dict(self.m.os.environ, {'COLAB_RELEASE_TAG': 'real', 'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}, clear=True):
            with self.assertRaises(ValueError):
                self.m.remote_guard()
        with patch.object(self.m.platform, 'system', return_value='Linux'), patch.dict(self.m.os.environ, {'COLAB_RELEASE_TAG': 'real', 'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1'}, clear=True):
            self.m.remote_guard()

    def test_frozen_settings_and_external_hash(self):
        self.m.check_settings(dict(self.m.SETTINGS))
        for updates in ({'max_new_tokens': 129}, {'attention': 'eager'}, {'seed': True}):
            with self.assertRaises(ValueError):
                self.m.check_settings(dict(self.m.SETTINGS, **updates))
        for value in ('main', 'A' * 64, '', 'f' * 63):
            with self.assertRaises(ValueError):
                self.m.check_digest(value)

    def test_failed_atomic_persist_preserves_prior_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            self.m.persist(path, {'status': 'preparing'})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                self.m.persist(path, {'bad': float('nan')})
            self.assertEqual(path.read_bytes(), before)

    def test_aggregate_language_separates_task_and_mind(self):
        mind = [{'language': 'en', 'valid': True, 'correct': False}, {'language': 'ar', 'valid': False, 'correct': False}]
        episodes = [{'language': 'en', 'metrics': {'success': True, 'model_calls': 4, 'model_errors': 0, 'unsafe_proposals': 0, 'unnecessary_calls': 0, 'recoveries': 1, 'action_errors': 0, 'mind_calls': 2, 'mind_correct': 1, 'mind_errors': 1}}]
        summary = self.m.summarize(episodes, mind)
        self.assertEqual(summary['en']['mind_correct'], 1)
        self.assertEqual(summary['en']['mind_semantic_accuracy'], .5)
        self.assertEqual(summary['en']['episodes_complete'], 1)
        self.assertEqual(summary['ar']['episodes'], 0)
        self.assertEqual(summary['ar']['mind_invalid'], 1)

    def seam(self, root, raw='```json\n{"kind":"next"}\n```', input_tokens=10, fail=False):
        class Tensor:
            shape = (1, 13)
            def __getitem__(self, key):
                self.trim = key[1].start
                return SimpleNamespace(shape=(1, 3))
        class Inputs(dict):
            input_ids = SimpleNamespace(shape=(1, input_tokens))
            def to(self, device):
                self.device = device
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'chat-template'
            def __call__(self, **kwargs):
                self.arguments = kwargs
                return Inputs()
            def batch_decode(self, tensor, **kwargs):
                return [raw]
        class Model:
            called = False
            def generate(self, **kwargs):
                self.called = True
                self.arguments = kwargs
                if fail:
                    raise RuntimeError('fake device failure')
                return Tensor()
        class Image:
            size = (800, 600)
            def convert(self, mode):
                return Image()
            def close(self):
                pass
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None), inference_mode=nullcontext)
        sink, model, processor = [], Model(), Processor()
        infer = self.m.QwenInfer(torch, model, processor, lambda _: Image(), root, sink,
                                sample=lambda _: {'device_used_bytes': 123})
        return infer, model, processor, sink

    def test_fake_model_seam_preserves_raw_without_parser_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            infer, model, processor, sink = self.seam(temp)
            raw = infer('mind', 'PUBLIC PROMPT ONLY', None)
            self.assertTrue(raw.startswith('```json'))
            self.assertNotIn('images', processor.arguments)
            self.assertEqual(processor.messages, [{'role': 'user', 'content': [{'type': 'text', 'text': 'PUBLIC PROMPT ONLY'}]}])
            self.assertEqual(model.arguments, {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128, 'max_time': 20})
            self.assertEqual(sink[0]['raw'], raw)
            self.assertEqual(sink[0]['output_tokens'], 3)
            self.assertIsNone(sink[0]['image_sha256'])
            self.assertEqual(sink[0]['memory_after']['device_used_bytes'], 123)

    def test_type_frame_binding_and_text_image_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / 'frame.png'
            frame.write_bytes(b'\x89PNG\r\n\x1a\n')
            infer, _, processor, sink = self.seam(root)
            infer('type', 'CURRENT FRAME', frame)
            self.assertEqual(len(processor.arguments['images']), 1)
            self.assertEqual(len(sink[0]['image_sha256']), 64)
            for component, image in [('mind', frame), ('type', None), ('oracle', None)]:
                with self.assertRaises(ValueError):
                    infer(component, 'PUBLIC', image)
            with self.assertRaises(ValueError):
                infer('type', 'PUBLIC', root.parent / 'outside.png')

    def test_expanded_token_budget_and_failure_record(self):
        with tempfile.TemporaryDirectory() as temp:
            infer, model, _, sink = self.seam(temp, input_tokens=4097)
            with self.assertRaises(ValueError):
                infer('mind', 'PUBLIC', None)
            self.assertFalse(model.called)
            self.assertEqual(sink[0]['error_type'], 'ValueError')
            infer, _, _, sink = self.seam(temp, fail=True)
            with self.assertRaises(RuntimeError):
                infer('mind', 'PUBLIC', None)
            self.assertEqual(sink[0]['error_type'], 'RuntimeError')
            self.assertGreaterEqual(sink[0]['wall_seconds'], 0)

    def test_expired_outer_deadline_prevents_further_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            infer, model, _, sink = self.seam(temp)
            infer.deadline_monotonic = 0
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    infer('mind', 'PUBLIC', None)
            self.assertFalse(model.called)
            self.assertEqual(sink, [])

    def test_sampler_failure_preserves_completed_raw_and_original_model_error(self):
        with tempfile.TemporaryDirectory() as temp:
            infer, model, _, sink = self.seam(temp, raw='RAW INVALID JSON')
            count = [0]
            def fail_after(_):
                count[0] += 1
                if count[0] > 1:
                    raise ValueError('unavailable telemetry')
                return {'device_used_bytes': 123}
            infer.sample = fail_after
            self.assertEqual(infer('mind', 'PUBLIC', None), 'RAW INVALID JSON')
            self.assertTrue(model.called)
            self.assertEqual(sink[0]['raw'], 'RAW INVALID JSON')
            self.assertIsNone(sink[0]['memory_after'])
            self.assertEqual(sink[0]['telemetry_errors']['memory_after'], 'ValueError')
            infer, _, _, sink = self.seam(temp, fail=True)
            infer.sample = lambda _: (_ for _ in ()).throw(ValueError('unavailable telemetry'))
            with self.assertRaisesRegex(RuntimeError, 'fake device failure'):
                infer('mind', 'PUBLIC', None)
            self.assertEqual(sink[0]['error_type'], 'RuntimeError')
            self.assertIsNone(sink[0]['memory_before'])
            self.assertIsNone(sink[0]['memory_after'])

    def test_final_sampler_failure_still_persists_blocked_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            result = {'status': 'blocked', 'error_type': 'RuntimeError', 'calls': [
                {'raw': 'PRESERVE', 'memory_before': {'device_used_bytes': 77}, 'memory_after': None}]}
            torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
            with patch.object(self.m, 'memory_sample', side_effect=RuntimeError('device unavailable')):
                self.m.finalize_result(result, path, torch, self.m.time.perf_counter(), self.m.time.process_time())
            saved = json.loads(path.read_text())
            self.assertEqual(saved['status'], 'blocked')
            self.assertEqual(saved['calls'][0]['raw'], 'PRESERVE')
            self.assertIsNone(saved['memory_final'])
            self.assertEqual(saved['telemetry_errors']['memory_final'], 'RuntimeError')
            self.assertEqual(saved['max_sampled_device_used_bytes'], 77)
            result = {'status': 'blocked', 'calls': []}
            torch.cuda.empty_cache = lambda: (_ for _ in ()).throw(RuntimeError('device unavailable'))
            with patch.object(self.m, 'memory_sample', side_effect=RuntimeError('device unavailable')):
                self.m.finalize_result(result, path, torch, self.m.time.perf_counter(), self.m.time.process_time())
            saved = json.loads(path.read_text())
            self.assertIsNone(saved['max_sampled_device_used_bytes'])
            self.assertEqual(saved['telemetry_errors']['empty_cache'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
