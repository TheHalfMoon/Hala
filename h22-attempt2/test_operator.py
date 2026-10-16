"""Stdlib-only adversarial operator checks; no ML or network execution."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Attempt2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load('attempt2_operator', 'h22-run-one-profile.py')
        cls.q = load('attempt2_qualify', 'h22-attempt2-qualify.py')
        cls.original = json.loads((HERE / 'h22-prospective-experiment.json').read_text())

    def manifest(self):
        return dict(copy.deepcopy(self.original), attempt=2, previous_attempt_results_retrieved=False,
            gpu_uuid='GPU-11111111-2222-3333-4444-555555555555')

    def test_fixed_study_rejects_budget_adapter_and_task_changes(self):
        m = self.m
        m.validate_manifest(self.manifest(), self.original)
        for key, value in [('mode_order', list(reversed(m.ORDER))), ('adapter_manifest_sha256', 'f' * 64),
                           ('source_commit', 'f' * 40), ('budget', {}), ('packages', {})]:
            bad = self.manifest()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_manifest(bad, self.original)
        for index in (-1, 6, True):
            with self.assertRaises(ValueError):
                m.profile_name(index)

    def test_immutable_urls_and_fetch_limits_reject_redirect_or_wrong_hash(self):
        m = self.m
        with self.assertRaises(ValueError):
            m.artifact_url('main', 'experiment.json')
        url = m.artifact_url('a' * 40, 'experiment.json')
        self.assertTrue(url.startswith(m.PREFIX + 'a' * 40 + '/h22-attempt2/'))
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def geturl(self): return url
            def read(self, limit): return b'x' * limit
        with patch.object(m.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaises(ValueError):
                m.fetch(url, m.sha(b'xx'), 1)
            with self.assertRaises(ValueError):
                m.fetch(url, '0' * 64, 10)
        class Redirect(Response):
            def geturl(self): return 'https://elsewhere.invalid/'
        with patch.object(m.urllib.request, 'urlopen', return_value=Redirect()):
            with self.assertRaises(ValueError):
                m.fetch(url, '0' * 64, 10)

    def test_clean_source_rejects_dirty_tree_before_qualification_imports(self):
        for module in (self.m, self.q):
            with patch.object(module.subprocess, 'check_output', return_value=module.COMMIT), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout=' M changed.py')):
                with self.assertRaises(ValueError):
                    module.clean_source(Path('/fake'))
        with patch.object(self.q, 'clean_source', side_effect=ValueError('dirty')), \
                patch.object(self.q.importlib.metadata, 'version') as packages:
            with self.assertRaises(ValueError):
                self.q.qualify('GPU-unknown', root=Path('/fake'))
            packages.assert_not_called()
        for module in (self.m, self.q):
            with patch.object(module.subprocess, 'check_output', return_value='f' * 40), \
                    patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
                with self.assertRaises(ValueError):
                    module.clean_source(Path('/fake'))

    def retention(self, *, index=0, experiment=None, omit=(), claimed_result=None, result_raw=None, terminal='failed'):
        m = self.m
        experiment = experiment or json.dumps(self.manifest()).encode()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('experiment.json', experiment)
            status = {'schema': 'hala.h22.single_profile.v1', 'index': index, 'name': m.profile_name(index),
                'source_commit': m.COMMIT, 'experiment_sha256': m.sha(experiment), 'status': terminal,
                'returncode': 0 if terminal == 'finished' else 1,
                'argv': ['python', '-m', 'evaluation.development_precision', '--mode', m.ORDER[index][0],
                         '--split', m.ORDER[index][1]]}
            if claimed_result is not None:
                status.update(result_sha256=claimed_result, result_status='measured')
            archive.writestr('profile-status.json', json.dumps(status))
            members = {'profile.log': b'raw failure', 'qualification.json': json.dumps({
                'attempt': 2, 'source_commit': m.COMMIT, 'source_freeze_sha256': m.FREEZE,
                'gpu_uuid': self.manifest()['gpu_uuid']}).encode(), 'operator-launcher.py': b'actual launcher'}
            if result_raw is not None:
                members['run/result.json'] = result_raw
            for name, value in members.items():
                if name not in omit:
                    archive.writestr(name, value)
        raw = stream.getvalue()
        receipt = {'schema': 'hala.h22.public_retention_receipt.v1', 'index': 0,
            'experiment_sha256': m.sha(experiment), 'archive_sha256': m.sha(raw),
            'archive_path': m.ARTIFACT + m.profile_name(0) + '-evidence.zip'}
        return experiment, raw, receipt

    def test_public_raw_zip_must_match_receipt_previous_index_and_experiment(self):
        m = self.m
        experiment, raw, receipt = self.retention()
        def check(receipt_value, zip_value=raw):
            with patch.object(m, 'fetch', side_effect=[json.dumps(receipt_value).encode(), zip_value]):
                return m.prior_retention(1, m.sha(experiment), 'a' * 40, 'b' * 64, m.sha(raw))
        self.assertEqual(check(receipt), receipt)
        for key, value in [('index', 1), ('experiment_sha256', 'f' * 64), ('archive_sha256', 'f' * 64),
                           ('archive_path', 'different.zip')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(dict(receipt, **{key: value}))
        _, wrong_index_zip, _ = self.retention(index=1)
        with self.assertRaises(ValueError):
            check(receipt, wrong_index_zip)
        _, wrong_experiment_zip, _ = self.retention(experiment=b'changed')
        with self.assertRaises(ValueError):
            check(receipt, wrong_experiment_zip)

    def test_retention_rejects_missing_raw_members_or_forged_result_claim(self):
        m = self.m
        for missing in ('profile.log', 'qualification.json', 'operator-launcher.py'):
            experiment, raw, receipt = self.retention(omit=(missing,))
            with self.subTest(missing=missing), patch.object(m, 'fetch', side_effect=[json.dumps(receipt).encode(), raw]):
                with self.assertRaises(ValueError):
                    m.prior_retention(1, m.sha(experiment), 'a' * 40, 'b' * 64, m.sha(raw))

    def test_measured_retention_checks_actual_result_status_and_mode(self):
        m = self.m
        result = {'status': 'measured', 'source_commit': m.COMMIT, 'split': 'dev',
            'precision_profile': {'mode': 'fp16', 'source_freeze_sha256': m.FREEZE},
            'gpu_uuid': self.manifest()['gpu_uuid']}
        def check(value):
            result_raw = json.dumps(value).encode()
            experiment, raw, receipt = self.retention(claimed_result=m.sha(result_raw),
                result_raw=result_raw, terminal='finished')
            with patch.object(m, 'fetch', side_effect=[json.dumps(receipt).encode(), raw]):
                return m.prior_retention(1, m.sha(experiment), 'a' * 40, 'b' * 64, m.sha(raw))
        self.assertEqual(check(result)['index'], 0)
        for key, value in [('status', 'blocked'), ('split', 'test'), ('source_commit', 'f' * 40),
                           ('gpu_uuid', 'GPU-other'), ('precision_profile', {'mode': 'nf4', 'source_freeze_sha256': m.FREEZE})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(dict(result, **{key: value}))
        for kwargs in ({'terminal': 'finished'}, {'claimed_result': 'f' * 64},
                       {'claimed_result': 'f' * 64, 'result_raw': b'{"status":"measured"}'}):
            experiment, raw, receipt = self.retention(**kwargs)
            with self.subTest(kwargs=kwargs), patch.object(m, 'fetch', side_effect=[json.dumps(receipt).encode(), raw]):
                with self.assertRaises(ValueError):
                    m.prior_retention(1, m.sha(experiment), 'a' * 40, 'b' * 64, m.sha(raw))

    def test_missing_retention_blocks_next_profile_before_output_or_process(self):
        self.drive(index=1, expect_missing=True)

    def test_single_failed_profile_is_zipped_and_cannot_autoadvance_or_rerun(self):
        self.drive(index=0)

    def test_timeout_preserves_terminal_status_and_raw_log_zip(self):
        self.drive(index=0, timeout=True)

    def drive(self, index, expect_missing=False, timeout=False):
        m = self.m
        manifest = self.manifest()
        raw = json.dumps(manifest).encode()
        original_raw = json.dumps(self.original).encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'h22-attempt2-qualification.json').write_text(json.dumps({
                'source_commit': m.COMMIT, 'source_freeze_sha256': m.FREEZE,
                'gpu_uuid': manifest['gpu_uuid'], 'attempt': 2}))
            def fetched(url, expected, limit):
                return original_raw if url == m.ORIGINAL_URL else raw
            with patch.object(m, 'clean_source'), patch.object(m, 'fetch', side_effect=fetched), \
                    patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1),
                        side_effect=m.subprocess.TimeoutExpired('fake profile', 3700) if timeout else None) as process, \
                    redirect_stdout(io.StringIO()):
                if expect_missing:
                    with self.assertRaises(ValueError):
                        m.run(index, 'a' * 40, m.sha(raw), root=root)
                    process.assert_not_called()
                    self.assertFalse((root / 'precision-attempt2').exists())
                else:
                    report = m.run(index, 'a' * 40, m.sha(raw), root=root)
                    self.assertEqual(report['status'], 'launcher_timeout' if timeout else 'failed')
                    self.assertEqual(process.call_count, 1)
                    self.assertEqual(process.call_args.kwargs['timeout'], 3700)
                    archive = root / 'precision-attempt2' / (m.profile_name(0) + '-evidence.zip')
                    with zipfile.ZipFile(archive) as retained:
                        self.assertIn('profile.log', retained.namelist())
                        self.assertIn('profile-status.json', retained.namelist())
                        self.assertEqual(m.sha(retained.read('experiment.json')), m.sha(raw))
                    with self.assertRaises(FileExistsError):
                        m.run(index, 'a' * 40, m.sha(raw), root=root)
                    self.assertEqual(process.call_count, 1)


if __name__ == '__main__':
    unittest.main()
