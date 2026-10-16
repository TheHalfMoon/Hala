"""CPU-only TEST restart guard, fixed retained DEV and complete raw-archive regressions."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from test_h22_attempt2_operator import load

HERE = Path(__file__).resolve().parent


class TestRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load('test_only_operator', 'h22-test-only-operator.py')
        cls.original = json.loads((HERE / 'h22-prospective-experiment.json').read_text())

    def manifest(self):
        value = json.loads((HERE / 'h22-test-restart-prospective-amendment.json').read_text())
        value['gpu_uuid'] = 'GPU-11111111-2222-3333-4444-555555555555'
        return value

    def test_only_three_fixed_test_modes_and_manifest_no_dev_or_changed_identity(self):
        m = self.m
        m.validate_manifest(self.manifest(), self.original)
        self.assertEqual(m.ORDER, [['fp16', 'test'], ['int8', 'test'], ['nf4', 'test']])
        for bad in (dict(self.manifest(), mode_order=[['fp16', 'dev']]),
                    dict(self.manifest(), gpu_uuid='PENDING_ACTUAL_QUALIFICATION'),
                    dict(self.manifest(), source_commit='f' * 40),
                    dict(self.manifest(), dev_experiment_sha256='f' * 64),
                    dict(self.manifest(), dev_retention=[])):
            with self.assertRaises(ValueError):
                m.validate_manifest(bad, self.original)
        with self.assertRaises(ValueError):
            m.profile_name(3)

    def test_all_three_original_dev_public_raw_prerequisites_are_fixed(self):
        m = self.m
        with patch.object(m, 'prior_retention', return_value={'retained': True}) as verify:
            self.assertEqual(len(m.dev_prerequisites()), 3)
            for call, row in zip(verify.call_args_list, m.DEV_RETENTION):
                self.assertEqual(call.args, (row['index'] + 1, m.DEV_SHA, row['commit'], row['receipt_sha256'], row['archive_sha256']))
                self.assertEqual(call.kwargs, {'group': 'dev'})
        with patch.object(m, 'prior_retention', side_effect=ValueError('missing raw')):
            with self.assertRaises(ValueError):
                m.dev_prerequisites()

    def test_original_actual_dev_zips_satisfy_unchanged_completeness_checks(self):
        m = self.m
        for row in m.DEV_RETENTION:
            index = row['index']
            name = 'profile-' + str(index) + '-' + '-'.join(m.DEV_ORDER[index])
            raw = (HERE / ('h22-attempt2-' + name + '-evidence.zip')).read_bytes()
            self.assertEqual(m.sha(raw), row['archive_sha256'])
            receipt = {'schema': 'hala.h22.public_retention_receipt.v1', 'index': index,
                'experiment_sha256': m.DEV_SHA, 'archive_sha256': row['archive_sha256'],
                'archive_path': 'h22-attempt2/' + name + '-evidence.zip'}
            with patch.object(m, 'fetch', side_effect=[json.dumps(receipt).encode(), raw]):
                self.assertEqual(m.prior_retention(index + 1, m.DEV_SHA, row['commit'],
                    row['receipt_sha256'], row['archive_sha256'], group='dev'), receipt)

    def test_wrong_source_stops_before_manifest_or_profile_process(self):
        m = self.m
        with patch.object(m, 'clean_source', side_effect=ValueError('wrong source')), patch.object(m, 'fetch') as fetch:
            with self.assertRaises(ValueError):
                m.run(0, 'a' * 40, 'b' * 64)
            fetch.assert_not_called()
        qualifier = load('test_restart_qualifier', 'h22-test-restart-qualify.py')
        with patch.object(qualifier, 'clean_source', side_effect=ValueError('dirty')), \
                patch.object(qualifier.importlib.metadata, 'version') as packages:
            with self.assertRaises(ValueError):
                qualifier.qualify('GPU-unknown')
            packages.assert_not_called()

    def test_first_test_runs_only_test_once_and_archives_dev_receipts(self):
        m = self.m
        manifest = self.manifest()
        raw = json.dumps(manifest).encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'h22-test-restart-qualification.json').write_text(json.dumps({'attempt': 3,
                'source_commit': m.COMMIT, 'source_freeze_sha256': m.FREEZE, 'gpu_uuid': manifest['gpu_uuid']}))
            with patch.object(m, 'clean_source'), patch.object(m, 'fetch', side_effect=lambda url, *args:
                    json.dumps(self.original).encode() if url == m.ORIGINAL_URL else raw), \
                    patch.object(m, 'dev_prerequisites', return_value=m.DEV_RETENTION), \
                    patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as process, \
                    redirect_stdout(io.StringIO()):
                report = m.run(0, 'a' * 40, m.sha(raw), root=root)
                self.assertEqual(report['status'], 'failed')
                argv = process.call_args.args[0]
                self.assertEqual(argv[argv.index('--split') + 1], 'test')
                self.assertEqual(process.call_count, 1)
                archive = root / 'precision-test-restart/profile-0-fp16-test-evidence.zip'
                with zipfile.ZipFile(archive) as retained:
                    self.assertIn('dev-retention-receipts.json', retained.namelist())
                with self.assertRaises(FileExistsError):
                    m.run(0, 'a' * 40, m.sha(raw), root=root)
                self.assertEqual(process.call_count, 1)

    def test_wrong_gpu_or_missing_old_dev_blocks_first_test_before_output(self):
        m = self.m
        for wrong_gpu in (False, True):
            with self.subTest(wrong_gpu=wrong_gpu), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self.manifest()
                raw = json.dumps(manifest).encode()
                qualification = {'attempt': 3, 'source_commit': m.COMMIT, 'source_freeze_sha256': m.FREEZE,
                    'gpu_uuid': 'GPU-other' if wrong_gpu else manifest['gpu_uuid']}
                (root / 'h22-test-restart-qualification.json').write_text(json.dumps(qualification))
                with patch.object(m, 'clean_source'), patch.object(m, 'fetch', side_effect=[raw, json.dumps(self.original).encode()]), \
                        patch.object(m, 'dev_prerequisites', side_effect=ValueError('missing old DEV')) as prior, \
                        patch.object(m.subprocess, 'run') as process:
                    with self.assertRaises(ValueError):
                        m.run(0, 'a' * 40, m.sha(raw), root=root)
                    process.assert_not_called()
                    self.assertFalse((root / 'precision-test-restart').exists())
                    self.assertEqual(prior.call_count, 0 if wrong_gpu else 1)

    def test_new_test_raw_gate_rejects_missing_members_and_forged_measured_result(self):
        m = self.m
        experiment = json.dumps(self.manifest()).encode()
        for missing in ('profile.log', 'qualification.json', 'operator-launcher.py', 'run/result.json'):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                values = {'experiment.json': experiment, 'profile.log': b'raw', 'operator-launcher.py': b'launcher',
                    'qualification.json': json.dumps({'attempt': 3, 'source_commit': m.COMMIT,
                        'source_freeze_sha256': m.FREEZE, 'gpu_uuid': self.manifest()['gpu_uuid']}),
                    'profile-status.json': json.dumps({'schema': 'hala.h22.single_profile.v1', 'index': 0,
                        'name': 'profile-0-fp16-test', 'source_commit': m.COMMIT, 'experiment_sha256': m.sha(experiment),
                        'status': 'finished', 'returncode': 0, 'result_status': 'measured', 'result_sha256': 'f' * 64,
                        'argv': ['python', '--mode', 'fp16', '--split', 'test']}),
                    'run/result.json': b'{"status":"measured"}'}
                for name, value in values.items():
                    if name != missing:
                        archive.writestr(name, value)
            raw = stream.getvalue()
            receipt = {'schema': 'hala.h22.public_retention_receipt.v1', 'index': 0,
                'experiment_sha256': m.sha(experiment), 'archive_sha256': m.sha(raw),
                'archive_path': 'h22-test-restart/profile-0-fp16-test-evidence.zip'}
            with self.subTest(missing=missing), patch.object(m, 'fetch', side_effect=[json.dumps(receipt).encode(), raw]):
                with self.assertRaises(ValueError):
                    m.prior_retention(1, m.sha(experiment), 'a' * 40, 'b' * 64, m.sha(raw))


if __name__ == '__main__':
    unittest.main()
