"""CPU contract tests. Fixtures are synthetic; they claim no model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from development.data import build_dataset, canonical
from development.freeze import freeze_bundle
from development.media import FONT_SHA256
from evaluation import development_runtime as runtime
from training.adapter_archive import create_bundle
from training.development_data import encode_row, load_rows
from training.scripts.train_development import (digest, preflight, SCHEDULE,
                                               validate_baseline, verified_adapter_path)


def fake_render(episode, state, path, font):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b'\x89PNG\r\n\x1a\nunit-fixture'
    path.write_bytes(raw)
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'font_sha256': FONT_SHA256,
            'pillow': '11.3.0', 'raqm': '0.10.1', 'platform': 'Linux', 'width': 800, 'height': 600}


class IDs:
    shape = (1, 3)
    def __init__(self, values):
        self.values = values
    def __getitem__(self, index):
        return self
    def tolist(self):
        return self.values


class Processor:
    def __init__(self, drift=False):
        self.calls, self.messages, self.drift = [], [], drift
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return 'prefix' if kwargs.get('add_generation_prompt') else 'full'
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        values = ([1, 2, 3] if kwargs['text'] == ['full'] else [9 if self.drift else 1, 2])
        return type('Inputs', (), {'input_ids': IDs(values)})()


class DevelopmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'data'
        with patch('development.freeze.render_frame', side_effect=fake_render):
            self.manifest = freeze_bundle(self.bundle, runtime.SETTINGS, 'fixture-font')
        self.manifest_sha = digest(self.bundle / 'manifest.json')

    def baseline_fixture(self):
        episodes = []
        for episode in build_dataset()['episodes']:
            if episode['split'] == 'dev':
                episodes.append({'episode': episode['id'], 'split': 'dev', 'language': episode['language'],
                    'calls': [{'raw': '{}'}], 'metrics': {'success': False, 'mind_correct': 0, 'mind_calls': 1}})
        mind = [dict(c, valid=False, correct=False) for c in build_dataset()['mind_cases']]
        return {'schema': 'hala.development.baseline.v1', 'status': 'measured', 'model_id': runtime.MODEL_ID,
            'model_revision': runtime.REVISION, 'settings': runtime.SETTINGS, 'source_commit': 'a'*40,
            'manifest_sha256': self.manifest_sha, 'data_file_sha256': self.manifest['files']['dataset.json'],
            'frozen_source': self.manifest['source'], 'training_runs': 0, 'adapter_loaded': False,
            'runtime_source_sha256': digest(runtime.__file__), 'split': 'dev', 'packages': runtime.VERSIONS,
            'episodes': episodes, 'mind_cases': mind, 'calls': [{'raw': '{}'}],
            'summary': runtime.summarize(episodes, mind)}

    def test_rederived_rows_exclude_reliance_and_private_state(self):
        rows, _ = load_rows(self.bundle, self.manifest_sha)
        authored = build_dataset()['rows']
        expected = [r for r in authored if r['component'] in ('mind', 'type')]
        self.assertEqual(len(authored), 136)
        self.assertEqual(len(rows), len(expected))
        self.assertEqual(len(rows), 116)
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in expected])
        self.assertEqual({r['language'] for r in rows}, {'en', 'ar'})
        self.assertFalse(any('state' in r or 'episode' in r for r in rows))
        self.assertTrue(all(r['image'] is None for r in rows if r['component'] == 'mind'))
        self.assertTrue(all(r['image'] is not None for r in rows if r['component'] == 'type'))

    def test_tampered_coherent_label_and_external_hash_rejected(self):
        path = self.bundle / 'training-rows.jsonl'
        records = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        records[1]['completion'] = '{}'
        path.write_bytes(b''.join(canonical(r)+b'\n' for r in records))
        with self.assertRaises(ValueError):
            load_rows(self.bundle, self.manifest_sha)
        # Even a self-consistent new manifest cannot bless changed authored labels.
        self.manifest['files']['training-rows.jsonl'] = digest(path)
        (self.bundle / 'manifest.json').write_bytes(canonical(self.manifest))
        with self.assertRaises(ValueError):
            load_rows(self.bundle, digest(self.bundle / 'manifest.json'))

    def test_mind_processor_text_only_and_prefix_mask(self):
        rows, _ = load_rows(self.bundle, self.manifest_sha)
        row = next(r for r in rows if r['component'] == 'mind')
        processor = Processor()
        _, labels = encode_row(row, processor, self.bundle, lambda _: self.fail('Mind opened image'))
        self.assertEqual(labels, [-100, -100, 3])
        self.assertTrue(all('images' not in c for c in processor.calls))
        self.assertEqual(processor.messages[0][0]['content'], [{'type': 'text', 'text': row['prompt']}])
        with self.assertRaises(ValueError):
            encode_row(row, Processor(drift=True), self.bundle, None)

    def test_type_media_digest_and_path_rejected_before_processor(self):
        rows, _ = load_rows(self.bundle, self.manifest_sha)
        row = next(r for r in rows if r['component'] == 'type')
        for replacement in ({'image_sha256': '0'*64}, {'image': '../escape.png'}, {'image': None}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                encode_row(dict(row, **replacement), Processor(), self.bundle, None)

    def test_baseline_identity_summary_and_deficit_enforced(self):
        baseline = self.baseline_fixture()
        validate_baseline(baseline, self.manifest, self.manifest_sha)
        for field, replacement in [('status', 'blocked'), ('adapter_loaded', True), ('training_runs', 1),
                                   ('runtime_source_sha256', '0'*64), ('manifest_sha256', '0'*64),
                                   ('model_revision', '0'*40), ('split', 'test'), ('episodes', [])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_baseline(dict(baseline, **{field: replacement}), self.manifest, self.manifest_sha)
        improved = copy.deepcopy(baseline)
        for e in improved['episodes']:
            e['metrics'].update(success=True, mind_correct=1)
        improved['summary'] = runtime.summarize(improved['episodes'], improved['mind_cases'])
        with self.assertRaisesRegex(ValueError, 'deficit'):
            validate_baseline(improved, self.manifest, self.manifest_sha)
        stale = copy.deepcopy(baseline)
        stale['summary']['all']['episodes_complete'] = 8
        with self.assertRaises(ValueError):
            validate_baseline(stale, self.manifest, self.manifest_sha)

    def test_roundtrip_identity_and_training_seam(self):
        baseline = self.root / 'baseline.json'
        baseline.write_text(json.dumps(self.baseline_fixture()), encoding='utf-8')
        identity = {'base_model': runtime.MODEL_ID, 'base_revision': runtime.REVISION,
                    'source_commit': 'b'*40, 'data_sha256': self.manifest['files']['dataset.json'],
                    'evaluation_sha256': digest(baseline)}
        empty = self.root / 'empty'
        empty.mkdir()
        probe = self.root / 'probe'
        probe_sha = create_bundle(empty, probe, identity, kind='pretraining_roundtrip')
        args = [self.bundle, self.manifest_sha, baseline, digest(baseline), 'b'*40,
                probe, probe_sha, 'c'*40, 'TheHalfMoon/Hala']
        rows, _, actual, baseline_commit = preflight(*args)
        self.assertEqual(actual, identity)
        self.assertEqual(baseline_commit, 'a'*40)
        self.assertEqual(len(rows), 116)
        with self.assertRaises(ValueError):
            verified_adapter_path(probe, identity, probe_sha)
        for index, wrong in [(3, '0'*64), (4, 'd'*40), (6, '0'*64), (7, 'main'), (8, 'other/repo')]:
            changed = list(args)
            changed[index] = wrong
            with self.subTest(index=index), self.assertRaises(ValueError):
                preflight(*changed)

    def test_schedule_is_single_authored_epoch(self):
        self.assertEqual(SCHEDULE['epochs'], 1)
        self.assertEqual(SCHEDULE['row_order'], 'authored')
        self.assertEqual(SCHEDULE['learning_rate'], 5e-5)
        self.assertEqual(SCHEDULE['initial_scale'], 1.0)

    def test_completed_step_persists_when_memory_sampling_raises(self):
        from training.scripts.train_development import record_completed_step
        result = {'metrics': []}
        def unavailable(_):
            raise RuntimeError('sensitive diagnostic must not be persisted')
        path = self.root / 'progress.json'
        metric = {'step': 1, 'loss': 0.5, 'changed_tensors': 56}
        record_completed_step(result, metric, object(), path, sample=unavailable)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['metrics'][0]['step'], 1)
        self.assertEqual(saved['metrics'][0]['changed_tensors'], 56)
        self.assertIsNone(saved['metrics'][0]['memory'])
        self.assertEqual(saved['metrics'][0]['telemetry_errors'], {'memory': 'RuntimeError'})
        self.assertNotIn('sensitive', path.read_text())

    def test_final_telemetry_failure_preserves_original_training_error(self):
        from training.scripts.train_development import finalize_training_result
        result = {'status': 'blocked', 'error_type': 'ValueError', 'metrics': []}
        path = self.root / 'result.json'
        def unavailable(_):
            raise OSError('private memory details')
        original = ValueError('original training failure')
        with self.assertRaises(ValueError) as caught:
            try:
                raise original
            finally:
                finalize_training_result(result, path, object(), 0, sample=unavailable)
        self.assertIs(caught.exception, original)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['error_type'], 'ValueError')
        self.assertIsNone(saved['memory_final'])
        self.assertEqual(saved['telemetry_errors'], {'memory_final': 'OSError'})
        self.assertNotIn('private memory', path.read_text())

    def test_oversized_baseline_rejected_before_hash_read(self):
        oversized = self.root / 'oversized.json'
        with oversized.open('wb') as handle:
            handle.truncate(16 * 1024 * 1024 + 1)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded read attempted')), \
                patch.object(Path, 'open', side_effect=AssertionError('oversized file opened')):
            with self.assertRaises(ValueError):
                preflight(self.bundle, self.manifest_sha, oversized, 'a'*64, 'b'*40,
                          self.root / 'missing-probe', 'c'*64, 'd'*40, 'TheHalfMoon/Hala')


if __name__ == '__main__':
    unittest.main()
