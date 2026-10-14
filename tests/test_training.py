import unittest
import re

from training.preflight import TrainingConfig, qualify_environment, validate_experiment
from training.scripts.train_adapter import LORA_TARGET_PATTERN


class TrainingPreflightTests(unittest.TestCase):
    def test_lora_regex_matches_language_projections_and_excludes_vision(self):
        for name in ('model.language_model.layers.0.self_attn.q_proj',
                     'model.language_model.layers.1.self_attn.v_proj'):
            self.assertIsNotNone(re.fullmatch(LORA_TARGET_PATTERN, name))
        for name in ('model.visual.blocks.0.attn.q_proj',
                     'model.language_model.layers.0.self_attn.k_proj'):
            self.assertIsNone(re.fullmatch(LORA_TARGET_PATTERN, name))

    def config(self, **updates):
        fields = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': 'a' * 40,
                  'precision': 'fp16', 'max_steps': 4, 'batch_size': 1,
                  'gradient_accumulation': 4, 'learning_rate': 0.0001,
                  'lora_rank': 8, 'platform': 'kaggle', 'download_remote_weights': True}
        fields.update(updates)
        return TrainingConfig.from_dict(fields)

    def test_exact_revision_and_unknown_fields(self):
        for kwargs in ({'revision': 'main'}, {'surprise': True}, {'max_steps': True},
                       {'max_steps': 0}, {'learning_rate': float('nan')},
                       {'platform': 'personal_pc'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config(**kwargs)

    def test_t4_bf16_is_rejected_and_gpus_not_pooled(self):
        probe = {'cuda_available': True, 'packages': {'torch': '2.5.1'},
                 'devices': [{'name': 'Tesla T4', 'capability': [7, 5],
                              'vram_bytes': 16 * 1024**3},
                             {'name': 'Tesla T4', 'capability': [7, 5],
                              'vram_bytes': 16 * 1024**3}]}
        result = qualify_environment(self.config(), probe)
        self.assertEqual(result['per_device_vram_bytes'], 16 * 1024**3)
        self.assertEqual(result['memory_pooled'], False)
        with self.assertRaises(ValueError):
            qualify_environment(self.config(precision='bf16'), probe)

    def test_preflight_requires_provenance_and_measured_deficit(self):
        manifest = {'schema': 'hala.training.manifest.v1', 'split': 'train',
                    'rights_cleared': True, 'source_kind': 'synthetic',
                    'dataset_sha256': 'b' * 64, 'families': ['form_train'],
                    'held_out_families': ['form_test']}
        baseline = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': 'a' * 40,
                    'evidence': 'real_inference', 'evaluated_tasks': 10,
                    'verified_success_rate': 0.5, 'deficit': 'wrong GUI target'}
        validate_experiment(self.config(), manifest, baseline)
        for modified in ({**manifest, 'source_kind': 'benchmark_trajectory'},
                         {**manifest, 'held_out_families': ['form_train']},
                         {**manifest, 'rights_cleared': False}):
            with self.assertRaises(ValueError):
                validate_experiment(self.config(), modified, baseline)
        with self.assertRaises(ValueError):
            validate_experiment(self.config(), manifest, {**baseline, 'evidence': 'mock'})


if __name__ == '__main__':
    unittest.main()
