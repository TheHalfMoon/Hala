import unittest
from unittest.mock import patch
from training.scripts.baseline import reject_kaggle_runtime


class ProviderIdentityTests(unittest.TestCase):
    def check_runtime(self, environment, root_exists, blocked):
        with patch.dict('os.environ', environment, clear=True), patch(
                'training.scripts.baseline.Path.exists', return_value=root_exists):
            if blocked:
                with self.assertRaisesRegex(ValueError, 'Kaggle'):
                    reject_kaggle_runtime()
            else:
                reject_kaggle_runtime()

    def test_observed_colab_image_has_kaggle_root(self):
        self.check_runtime({'COLAB_RELEASE_TAG': 'observed'}, True, False)

    def test_kaggle_marker_wins_over_colab_marker(self):
        self.check_runtime({'COLAB_RELEASE_TAG': 'present',
                            'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}, True, True)

    def test_unknown_kaggle_root_stays_blocked(self):
        self.check_runtime({}, True, True)

    def test_generic_linux_without_root_is_allowed(self):
        self.check_runtime({}, False, False)
