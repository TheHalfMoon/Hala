import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.factory import RIGHTS, generate_manifest
from training.artifacts import load_sft_artifact, sha256_file
from training.checkpoints import publish_checkpoint, load_checkpoint, completed_checkpoint_result
from training.masking import completion_labels
from training.scripts.convert_sft import convert
from training.scripts.train_adapter import train


class TrainingIntegrityTests(unittest.TestCase):
    def artifact(self, directory):
        image = directory / 'images/a.png'
        image.parent.mkdir()
        from PIL import Image
        Image.new('RGB', (2, 2), 'white').save(image, format='PNG')
        row = {'id': 'case-a', 'split': 'train', 'family': 'workflow-save',
               'language': 'en', 'prompt': 'Click Save',
               'completion': '{"type":"semantic_invoke","element_ref":"target"}',
               'image_path': 'images/a.png', 'image_sha256': sha256_file(image)}
        examples = directory / 'examples.jsonl'
        examples.write_text(json.dumps(row) + '\n', encoding='utf-8')
        manifest = {'schema': 'hala.training.manifest.v1', 'split': 'train',
                    'rights_cleared': True, 'source_kind': 'synthetic',
                    'dataset_sha256': sha256_file(examples),
                    'families': ['workflow-save'], 'held_out_families': ['workflow-export'],
                    'token_budget': 4096, 'media': [
                        {'path': row['image_path'], 'sha256': row['image_sha256'],
                         'family': row['family'], 'provenance': dict(RIGHTS)}],
                    'arabic_image_training': {'enabled': False, 'raqm_qualified': False}}
        return examples, image, row, manifest

    def test_media_substitution_is_rejected_before_runtime_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            examples, image, _, manifest = self.artifact(Path(temp))
            self.assertEqual(len(load_sft_artifact(examples, manifest)), 1)
            image.write_bytes(b'different image, unchanged JSONL digest')
            with self.assertRaisesRegex(ValueError, 'digest'):
                load_sft_artifact(examples, manifest)

    def test_record_schema_path_rights_and_budget_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            examples, _, row, manifest = self.artifact(directory)
            for update in ({'split': 'test'}, {'family': 'workflow-export'},
                           {'prompt': ''}, {'prompt': 'x' * 32769},
                           {'completion': 'x' * 4097}, {'image_path': '../escape.png'},
                           {'image_path': 'images/../images/a.png'},
                           {'image_path': 'C:/image.png'}, {'extra': 'authority'},
                           {'image_sha256': '0' * 64}, {'language': 'ar'}):
                changed = {**row, **update}
                examples.write_text(json.dumps(changed) + '\n', encoding='utf-8')
                bound = {**manifest, 'dataset_sha256': sha256_file(examples)}
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_sft_artifact(examples, bound)
            examples.write_text(json.dumps(row) + '\n', encoding='utf-8')
            bound = {**manifest, 'dataset_sha256': sha256_file(examples)}
            for changed in ({**bound, 'token_budget': True},
                            {**bound, 'token_budget': 1000000},
                            {**bound, 'media': []},
                            {**bound, 'media': [{**bound['media'][0], 'provenance': {'source': 'benchmark'}}]}):
                with self.assertRaises(ValueError):
                    load_sft_artifact(examples, changed)

    def test_jsonl_duplicate_keys_and_duplicate_record_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            examples, _, row, manifest = self.artifact(Path(temp))
            for payload in ('{"id":"case-a","id":"case-b"}\n',
                            (json.dumps(row) + '\n') * 2):
                examples.write_text(payload, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_sft_artifact(examples, {**manifest, 'dataset_sha256': sha256_file(examples)})

    def test_completion_mask_excludes_all_prompt_and_vision_tokens(self):
        self.assertEqual(completion_labels([10, 500, 501, 11, 20, 21],
                                           [10, 500, 501, 11], 16),
                         [-100, -100, -100, -100, 20, 21])
        for full, prefix, budget in (([1, 2], [1, 2], 16),
                                     ([1, 2, 3], [1, 9], 16),
                                     ([1, 2, 3], [], 16),
                                     ([1, 2, 3], [1], 2)):
            with self.subTest(full=full, prefix=prefix), self.assertRaises(ValueError):
                completion_labels(full, prefix, budget)

    def write_checkpoint(self, stage, interrupted=False):
        adapter = stage / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter.safetensors').write_bytes(b'adapter weights')
        if interrupted:
            raise RuntimeError('Simulated optimizer write interruption')
        (stage / 'resume.pt').write_bytes(b'optimizer and scaler')

    def state(self, step=1):
        return {'config_sha256': 'a' * 64, 'dataset_sha256': 'b' * 64,
                'manifest_sha256': 'c' * 64, 'step': step, 'max_steps': 2,
                'metrics': [{'step': index, 'loss': 0.5, 'seconds': 1.0}
                            for index in range(1, step + 1)],
                'changed_parameter_tensors': 2, 'peak_vram_bytes': 100,
                'runtime': {'cuda_available': True}}

    def test_interrupted_write_preserves_previous_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            first = publish_checkpoint(output, self.state(), self.write_checkpoint)
            with self.assertRaises(RuntimeError):
                publish_checkpoint(output, self.state(2),
                                   lambda stage: self.write_checkpoint(stage, interrupted=True))
            version, state = load_checkpoint(output, self.state())
            self.assertEqual(version, first)
            self.assertEqual(state['step'], 1)
            self.assertEqual(state['metrics'], self.state()['metrics'])
            with self.assertRaises(ValueError):
                publish_checkpoint(output, self.state(), self.write_checkpoint)

    def test_interruption_after_version_rename_before_pointer_preserves_old_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            first = publish_checkpoint(output, self.state(), self.write_checkpoint)
            from training.checkpoints import atomic_json
            def fail_pointer(path, value):
                if Path(path).name == 'latest.json':
                    raise RuntimeError('Simulated interruption before pointer publish')
                atomic_json(path, value)
            with patch('training.checkpoints.atomic_json', side_effect=fail_pointer):
                with self.assertRaises(RuntimeError):
                    publish_checkpoint(output, self.state(2), self.write_checkpoint)
            self.assertEqual(load_checkpoint(output, self.state())[0], first)
            # Orphaned complete versions cannot block a later successful publication.
            second = publish_checkpoint(output, self.state(2), self.write_checkpoint)
            self.assertEqual(load_checkpoint(output, self.state(2))[0], second)

    def test_corruption_missing_extra_files_and_identity_change_reject_resume(self):
        for modification in ('corrupt', 'missing', 'extra', 'nested_metadata', 'identity', 'metadata'):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                version = publish_checkpoint(output, self.state(), self.write_checkpoint)
                expected = self.state()
                if modification == 'corrupt':
                    (version / 'adapter/adapter.safetensors').write_bytes(b'other')
                elif modification == 'missing':
                    (version / 'resume.pt').unlink()
                elif modification == 'extra':
                    (version / 'unbound.bin').write_bytes(b'extra')
                elif modification == 'nested_metadata':
                    (version / 'adapter/checkpoint.json').write_bytes(b'unbound')
                elif modification == 'metadata':
                    metadata = json.loads((version / 'checkpoint.json').read_text(encoding='utf-8'))
                    metadata['state']['changed_parameter_tensors'] = 99
                    (version / 'checkpoint.json').write_text(json.dumps(metadata), encoding='utf-8')
                else:
                    expected['manifest_sha256'] = 'd' * 64
                with self.assertRaises(ValueError):
                    load_checkpoint(output, expected)

    def test_completed_resume_finalizes_with_previous_metrics_and_update_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            publish_checkpoint(output, self.state(2), self.write_checkpoint)
            version, state = load_checkpoint(output, self.state(2))
            result = completed_checkpoint_result(version, state, {'max_steps': 2})
            self.assertEqual(result['metrics'], self.state(2)['metrics'])
            self.assertEqual(result['changed_parameter_tensors'], 2)
            self.assertFalse(result['adapter_qualified'])
            self.assertTrue(result['evaluation_required'])

    def write_runner_inputs(self, directory, manifest):
        config = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': 'a' * 40,
                  'precision': 'fp16', 'max_steps': 2, 'batch_size': 1,
                  'gradient_accumulation': 4, 'learning_rate': 0.0001,
                  'lora_rank': 8, 'platform': 'colab', 'download_remote_weights': True}
        baseline = {'model_id': config['model_id'], 'revision': config['revision'],
                    'evidence': 'real_inference', 'evaluated_tasks': 10,
                    'verified_success_rate': 0.5, 'deficit': 'Synthetic TEST INPUT ONLY'}
        paths = [directory / name for name in ('config.json', 'manifest.json', 'baseline.json')]
        for path, value in zip(paths, (config, manifest, baseline)):
            path.write_text(json.dumps(value), encoding='utf-8')
        return config, paths

    def test_runner_refuses_changed_media_before_runtime_or_model_import(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            examples, image, _, manifest = self.artifact(directory)
            _, paths = self.write_runner_inputs(directory, manifest)
            image.write_bytes(b'replaced')
            with patch('training.scripts.train_adapter.platform.system', return_value='Linux'), \
                    patch.dict('os.environ', {'HALA_REMOTE_TRAINING_ALLOWED': '1'}), \
                    patch('training.scripts.train_adapter.probe') as probe:
                with self.assertRaisesRegex(ValueError, 'digest'):
                    train(paths[0], examples, paths[1], paths[2], directory / 'output')
                probe.assert_not_called()

    def test_runner_completed_resume_finalizes_without_runtime_or_model_import(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            examples, _, _, manifest = self.artifact(directory)
            config, paths = self.write_runner_inputs(directory, manifest)
            state = self.state(2)
            state.update(config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                         dataset_sha256=manifest['dataset_sha256'], manifest_sha256=sha256_file(paths[1]))
            output = directory / 'output'
            publish_checkpoint(output, state, self.write_checkpoint)
            with patch('training.scripts.train_adapter.platform.system', return_value='Linux'), \
                    patch.dict('os.environ', {'HALA_REMOTE_TRAINING_ALLOWED': '1'}), \
                    patch('training.scripts.train_adapter.probe') as probe:
                result = train(paths[0], examples, paths[1], paths[2], output, resume=True)
                probe.assert_not_called()
            self.assertEqual(result['metrics'], state['metrics'])
            self.assertEqual(json.loads((output / 'training-result.json').read_text(encoding='utf-8')), result)

    def test_converter_only_exports_original_train_images_and_logical_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = generate_manifest()
            destination = Path(temp) / 'sft'
            examples, manifest_path = convert(manifest, destination, raqm_qualified=False)
            bound = json.loads(manifest_path.read_text(encoding='utf-8'))
            records = load_sft_artifact(examples, bound)
            expected = [case for case in manifest['tasks'] if case['split'] == 'train' and case['language'] == 'en']
            self.assertEqual(len(records), len(expected))
            self.assertEqual(bound['source_sha256'], __import__('hashlib').sha256(
                __import__('data.factory', fromlist=['manifest_bytes']).manifest_bytes(manifest)).hexdigest())
            self.assertEqual(set(bound['held_out_families']),
                             {case['family_id'] for case in manifest['tasks'] if case['split'] != 'train'})
            self.assertFalse(bound['arabic_image_training']['enabled'])
            for row in records:
                completion = json.loads(row['completion'])
                self.assertIn(completion['type'], ('click', 'semantic_invoke', 'set_value', 'scroll', 'observe'))
                self.assertFalse(set(completion) & {'metadata', 'confidence', 'success', 'receipt'})
                self.assertEqual((destination / row['image_path']).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(ValueError):
                convert(manifest, destination, raqm_qualified=False)
            poisoned = copy.deepcopy(manifest)
            poisoned['tasks'][0]['provenance']['benchmark_evaluation'] = True
            with self.assertRaises(ValueError):
                convert(poisoned, Path(temp) / 'poisoned', raqm_qualified=False)


if __name__ == '__main__':
    unittest.main()
