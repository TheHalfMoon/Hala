"""CPU admission tests do not establish a real provider's GPU eligibility."""
import builtins
from contextlib import ExitStack
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from evaluation import development_runtime as runtime


class DevelopmentProviderTests(unittest.TestCase):
    def environment(self, values, *, system='Linux', directory=True):
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, values, clear=True))
        stack.enter_context(patch.object(runtime.platform, 'system', return_value=system))
        stack.enter_context(patch.object(Path, 'is_dir', return_value=directory))
        return stack

    def test_default_colab_and_explicit_kaggle_provenance(self):
        with self.environment({'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'release'}):
            self.assertEqual(runtime.evaluation_guard(), {'name': 'colab', 'colab_release_tag': 'release'})
        for mode in ('Interactive', 'Batch'):
            with self.subTest(mode=mode), self.environment({
                    'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'KAGGLE_KERNEL_RUN_TYPE': mode}):
                self.assertEqual(runtime.evaluation_guard('kaggle'), {
                    'name': 'kaggle', 'kernel_run_type': mode, 'working_directory': '/kaggle/working'})
                with self.assertRaises(ValueError):
                    runtime.remote_guard()  # Training remains Colab-only.
                with self.assertRaises(ValueError):
                    runtime.evaluation_guard()  # Kaggle is never the default.

    def test_invalid_requests_fail_before_any_ml_import_or_adapter_validation(self):
        valid = {'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}
        cases = [
            ('unknown', valid, 'Linux', True),
            (None, valid, 'Linux', True),
            ('Kaggle', valid, 'Linux', True),
            ('kaggle', {}, 'Linux', True),
            ('kaggle', dict(valid, HALA_REMOTE_DEVELOPMENT_ALLOWED='0'), 'Linux', True),
            ('kaggle', dict(valid, KAGGLE_KERNEL_RUN_TYPE='Unknown'), 'Linux', True),
            ('kaggle', dict(valid, KAGGLE_KERNEL_RUN_TYPE=''), 'Linux', True),
            ('kaggle', dict(valid, COLAB_RELEASE_TAG='release'), 'Linux', True),
            ('kaggle', dict(valid, COLAB_RELEASE_TAG=''), 'Linux', True),
            ('kaggle', valid, 'Windows', True),
            ('kaggle', valid, 'Linux', False),
            ('colab', valid, 'Linux', True),
            ('colab', {'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1'}, 'Linux', True),
            ('colab', dict(valid, COLAB_RELEASE_TAG='release'), 'Linux', True),
            ('colab', {'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'release',
                       'KAGGLE_KERNEL_RUN_TYPE': ''}, 'Linux', True),
            ('colab', {'COLAB_RELEASE_TAG': 'release'}, 'Linux', True),
            ('colab', {'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'release'}, 'Windows', True),
        ]
        original_import = builtins.__import__
        for provider, environment, system, directory in cases:
            for candidate in (False, True):
                imported = []
                def guarded(name, *args, **kwargs):
                    if name.split('.')[0] in ('torch', 'transformers', 'PIL', 'peft', 'safetensors') or name == 'evaluation.development_adapter':
                        imported.append(name)
                        raise AssertionError('Provider rejected after protected import: ' + name)
                    return original_import(name, *args, **kwargs)
                kwargs = {'adapter': 'missing', 'adapter_manifest_sha256': 'c' * 64,
                          'adapter_identity': {}} if candidate else {}
                with self.subTest(provider=provider, environment=environment, system=system,
                                  directory=directory, candidate=candidate), self.environment(
                        environment, system=system, directory=directory), patch('builtins.__import__', side_effect=guarded):
                    with self.assertRaises(ValueError):
                        runtime.run('missing', 'b' * 64, 'output', 'font', 'a' * 40,
                                    provider=provider, **kwargs)
                    self.assertEqual(imported, [])

    def test_training_guard_accepts_colab_without_evaluation_provider(self):
        with self.environment({'HALA_REMOTE_DEVELOPMENT_ALLOWED': '1', 'COLAB_RELEASE_TAG': 'release'}):
            runtime.remote_guard()

    def test_cli_defaults_to_colab_and_requires_explicit_known_provider(self):
        argv = ['runtime', '--bundle', 'bundle', '--manifest-sha256', 'b' * 64,
                '--output', 'output', '--font-path', 'font', '--source-commit', 'a' * 40]
        with patch.object(runtime, 'run', return_value={'status': 'fake', 'summary': {}}) as execute:
            for flags, expected in (([], 'colab'), (['--provider', 'kaggle'], 'kaggle')):
                with self.subTest(flags=flags), patch.object(sys, 'argv', argv + flags), redirect_stdout(io.StringIO()):
                    runtime.main()
                    self.assertEqual(execute.call_args.kwargs['provider'], expected)
            execute.reset_mock()
            with patch.object(sys, 'argv', argv + ['--provider', 'linux']), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runtime.main()
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
