import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.scripts.qualify_lora import compare_values, compare_checkpoints, qualify, validate_smoke_config
from training.scripts.train_adapter import QualificationInterrupted, interrupt_after_checkpoint


class QualificationTests(unittest.TestCase):
    def test_loss_trajectory_mismatch_blocks_qualification_but_timing_does_not(self):
        state = {'step': 2, 'max_steps': 2, 'config_sha256': 'a' * 64,
                 'dataset_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64,
                 'metrics': [{'step': 1, 'loss': 1.5, 'seconds': 2.0},
                             {'step': 2, 'loss': 1.4, 'seconds': 2.0}]}
        fake_torch = SimpleNamespace(load=lambda path, **kwargs: {'optimizer': [1], 'torch_rng': [2]})
        fake_safe = SimpleNamespace(load_file=lambda path: {'adapter': [0.25]})
        for loss, expected in ((1.4, True), (2.4, False)):
            other = {**state, 'metrics': [state['metrics'][0], {'step': 2, 'loss': loss, 'seconds': 9.0}]}
            with patch.dict('sys.modules', {'torch': fake_torch, 'safetensors.torch': fake_safe}), \
                    patch('training.scripts.qualify_lora.snapshot_state', side_effect=
                          [(Path('continuous'), state), (Path('resumed'), other)]):
                compared = compare_checkpoints('continuous', 'resumed')
            self.assertEqual(compared['within_tolerance'], expected)
    def test_comparison_checks_structure_values_and_nonfinite(self):
        self.assertTrue(compare_values({'optimizer': [1, 0.25]}, {'optimizer': [1, 0.25]})['exact'])
        for other in ({'optimizer': [1, 0.26]}, {'optimizer': [1]}, {'rng': [1, 0.25]}):
            self.assertFalse(compare_values({'optimizer': [1, 0.25]}, other)['within_tolerance'])
        self.assertFalse(compare_values(float('nan'), float('nan'))['within_tolerance'])

    def test_interruption_only_after_published_step_one(self):
        calls = []
        def publish():
            calls.append('published')
            return Path('paired-version')
        version = publish()
        with self.assertRaises(QualificationInterrupted) as stopped:
            interrupt_after_checkpoint(1, 1, version)
        self.assertEqual(calls, ['published'])
        self.assertEqual(stopped.exception.step, 1)
        interrupt_after_checkpoint(None, 1, version)
        interrupt_after_checkpoint(1, 2, version)

    def test_smoke_bound_rejects_large_or_changed_run(self):
        config = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct',
                  'revision': '89644892e4d85e24eaac8bacfd4f463576704203',
                  'precision': 'fp16', 'max_steps': 2, 'batch_size': 1,
                  'gradient_accumulation': 1, 'lora_rank': 4,
                  'learning_rate': 0.00005, 'platform': 'colab',
                  'download_remote_weights': True}
        validate_smoke_config(config)
        for key, value in (('max_steps', 3), ('gradient_accumulation', 2),
                           ('revision', 'a' * 40), ('precision', 'bf16')):
            with self.assertRaises(ValueError):
                validate_smoke_config({**config, key: value})

    def test_saved_identity_and_optimizer_rng_mismatches_fail(self):
        state = {'step': 2, 'max_steps': 2, 'config_sha256': 'a' * 64,
                 'dataset_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64,
                 'metrics': [{'step': 1, 'loss': 1.5, 'seconds': 1.0},
                             {'step': 2, 'loss': 1.4, 'seconds': 1.0}]}
        fake_torch = SimpleNamespace(load=lambda path, **kwargs:
            {'optimizer': {'value': 1 if 'continuous' in str(path) else 2},
             'scaler': {'scale': 1024.0}, 'torch_rng': [1], 'cuda_rng': [[2]]})
        fake_safetensors = SimpleNamespace(load_file=lambda path: {'adapter': [0.25]})
        with patch.dict('sys.modules', {'torch': fake_torch, 'safetensors.torch': fake_safetensors}), \
                patch('training.scripts.qualify_lora.snapshot_state', side_effect=
                      [(Path('continuous'), state), (Path('resumed'), state)]):
            compared = compare_checkpoints('continuous', 'resumed')
        self.assertFalse(compared['within_tolerance'])
        self.assertEqual(compared['components']['optimizer']['mismatch_paths'], ['state.value'])
        with patch.dict('sys.modules', {'torch': fake_torch, 'safetensors.torch': fake_safetensors}), \
                patch('training.scripts.qualify_lora.snapshot_state', side_effect=
                      [(Path('continuous'), state), (Path('resumed'), {**state, 'dataset_sha256': 'd' * 64})]):
            with self.assertRaisesRegex(ValueError, 'identity'):
                compare_checkpoints('continuous', 'resumed')

    def test_local_runtime_and_input_mutation_reject_before_resume(self):
        with patch('training.scripts.qualify_lora.platform.system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'remote Linux'):
                qualify('missing', 'missing', 'missing', 'missing', 'missing')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = Path(__file__).parents[1] / 'experiments/runtime/colab-lora-smoke-config.json'
            for name in ('examples', 'manifest', 'baseline'):
                (directory / name).write_text('{}', encoding='utf-8')
            calls = []
            def mutate(*args, **kwargs):
                calls.append('continuous')
                (directory / 'baseline').write_text('{"changed": true}', encoding='utf-8')
            with patch('training.scripts.qualify_lora.release_runtime'):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    qualify(config, directory / 'examples', directory / 'manifest',
                            directory / 'baseline', directory / 'out', trainer=mutate)
            self.assertEqual(calls, ['continuous'])

    def test_orchestrator_requires_partial_snapshot_then_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = Path(__file__).parents[1] / 'experiments/runtime/colab-lora-smoke-config.json'
            for name in ('examples', 'manifest', 'baseline'):
                (directory / name).write_text('{}', encoding='utf-8')
            calls = []
            def trainer(*args, **kwargs):
                calls.append((kwargs.get('resume', False), kwargs.get('stop_after_step')))
                if kwargs.get('stop_after_step') == 1:
                    raise QualificationInterrupted(1, Path(args[4]) / 'first')
                return {'runtime': {}, 'metrics': []}
            partial = {'step': 1, 'max_steps': 2, 'config_sha256': 'a' * 64,
                       'dataset_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64}
            with patch('training.scripts.qualify_lora.snapshot_state', return_value=(directory, partial)), \
                    patch('training.scripts.qualify_lora.checkpoint_evidence', return_value={}), \
                    patch('training.scripts.qualify_lora.compare_checkpoints', return_value={'within_tolerance': True}), \
                    patch('training.scripts.qualify_lora.release_runtime'):
                report = qualify(config, directory / 'examples', directory / 'manifest',
                                 directory / 'baseline', directory / 'out', trainer=trainer)
            self.assertEqual(calls, [(False, None), (False, 1), (True, None)])
            self.assertEqual(report['evidence'], 'cpu_control_test_unqualified')
            self.assertFalse(report['runtime_qualified'])
            self.assertTrue((directory / 'out/qualification-result.json').exists())
            with patch('training.scripts.qualify_lora.snapshot_state', return_value=(directory, {**partial, 'step': 2})), \
                    patch('training.scripts.qualify_lora.release_runtime'):
                with self.assertRaisesRegex(ValueError, 'partial'):
                    qualify(config, directory / 'examples', directory / 'manifest',
                            directory / 'baseline', directory / 'other', trainer=trainer)


if __name__ == '__main__':
    unittest.main()
