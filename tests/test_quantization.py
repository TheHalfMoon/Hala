import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.quantization import quantization_config, paired, verify_media, remote_guard, verify_sources
from evaluation.quantization_runner import run


class QuantizationTests(unittest.TestCase):
    def test_prespecified_quantizers_disable_offload(self):
        self.assertEqual(quantization_config('int8')['llm_int8_threshold'], 6.0)
        self.assertFalse(quantization_config('int8')['llm_int8_enable_fp32_cpu_offload'])
        self.assertEqual(quantization_config('nf4')['bnb_4bit_quant_type'], 'nf4')
        with self.assertRaises(ValueError):
            quantization_config('fp16')

    def test_paired_counts_and_reject_missing_duplicate_or_media_change(self):
        base = [{'id': str(i), 'language': 'ar', 'family': 'a', 'image_sha256': 'x',
                 'hit': i < 2, 'schema_valid': True, 'call_seconds': 1.0} for i in range(4)]
        quant = copy.deepcopy(base)
        quant[1]['hit'] = False
        quant[2]['hit'] = True
        report = paired(base, quant)
        self.assertEqual(report['overall']['hit_pairs'],
                         {'both_hit': 1, 'lost_hit': 1, 'gained_hit': 1, 'neither_hit': 1})
        self.assertFalse(report['adoption_qualified'])
        for bad in (quant[:-1], quant + [quant[0]]):
            with self.assertRaises(ValueError):
                paired(base, bad)
        quant[0]['image_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            paired(base, quant)

    def test_guard_blocks_local_and_external_cache_before_imports(self):
        with patch('evaluation.quantization.platform.system', return_value='Windows'):
            with self.assertRaises(ValueError):
                remote_guard()
        with patch('evaluation.quantization.platform.system', return_value='Linux'), patch.dict(
                'os.environ', {'HALA_REMOTE_QUANT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'x',
                               'HF_HOME': '/content/drive/cache'}, clear=True):
            with self.assertRaises(ValueError):
                remote_guard()

    def test_colab_guard_allows_incidental_kaggle_directory(self):
        with patch('evaluation.quantization.platform.system', return_value='Linux'), patch.dict(
                'os.environ', {'HALA_REMOTE_QUANT_ALLOWED': '1',
                               'COLAB_RELEASE_TAG': 'release'}, clear=True), patch(
                'evaluation.quantization.Path.exists', return_value=True), patch(
                'evaluation.quantization.Path.is_symlink', return_value=False):
            remote_guard()

    def test_colab_guard_still_rejects_kaggle_marker_and_missing_colab_opt_in(self):
        environments = (
            {'HALA_REMOTE_QUANT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'release',
             'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'},
            {'HALA_REMOTE_QUANT_ALLOWED': '1'},
            {'COLAB_RELEASE_TAG': 'release'},
        )
        with patch('evaluation.quantization.platform.system', return_value='Linux'), patch(
                'evaluation.quantization.Path.exists', return_value=True), patch(
                'evaluation.quantization.Path.is_symlink', return_value=False):
            for environment in environments:
                with self.subTest(environment=environment), patch.dict(
                        'os.environ', environment, clear=True):
                    with self.assertRaises(ValueError):
                        remote_guard()

    def test_media_mapping_rejects_traversal_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = {'cases': [{'id': 'a', 'image_sha256': 'x'}],
                        'render_provenance': {'images': [{'id': 'a', 'path': '../escape.png',
                                                        'png_sha256': 'x'}]}}
            with self.assertRaises(ValueError):
                verify_media(baseline, Path(directory))
            baseline['render_provenance']['images'][0]['path'] = 'a.png'
            with self.assertRaises(ValueError):
                verify_media(baseline, Path(directory))

    def test_new_source_freeze_and_immutable_h14_verified(self):
        verify_sources()
        with self.assertRaises(ValueError):
            verify_sources('0' * 64)

    def test_package_failure_persists_blocked_without_importing_frameworks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result'
            with patch('evaluation.quantization_runner.remote_guard'), patch(
                    'evaluation.quantization_runner.verify_sources', return_value=(
                        {'packages': {'missing-hala-package': '1'}, 'baseline_sha256': 'x'},
                        {'cases': []}, {'manifest_sha256': 'x', 'generation': {}})), patch(
                    'evaluation.quantization_runner.verify_media', return_value=[]), patch(
                    'evaluation.quantization_runner.Path', wraps=Path) as paths, patch(
                    'evaluation.quantization_runner.signal.signal'), patch(
                    'evaluation.quantization_runner.signal.alarm', create=True), patch(
                    'evaluation.quantization_runner.signal.SIGALRM', 14, create=True):
                # Exercise the real failure/persistence path using a temporary output;
                # the Linux /tmp containment rule has its own native guard coverage.
                paths.return_value = output
                with self.assertRaises(Exception):
                    run('int8', directory, str(output), 'x')
                saved = json.loads((output / 'result.json').read_text())
                self.assertEqual(saved['status'], 'blocked')
                self.assertEqual(saved['evaluated_count'], 0)
                self.assertNotIn('summary', saved)


if __name__ == '__main__':
    unittest.main()
