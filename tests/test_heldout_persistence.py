import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evaluation import heldout_runner as runner


class PersistenceTests(unittest.TestCase):
    def test_failed_publication_preserves_prior_result_and_image_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('result.json', 'image-freeze.json'):
                path = Path(temp) / name
                prior = {'status': 'running', 'cases': [{'id': 'first'}]}
                runner.atomic_json(path, prior)
                with patch('training.checkpoints.os.replace', side_effect=OSError('interrupted')):
                    with self.assertRaises(OSError):
                        runner.atomic_json(path, {'status': 'running', 'cases': [{'id': 'first'}, {'id': 'second'}]})
                self.assertEqual(json.loads(path.read_text()), prior)
