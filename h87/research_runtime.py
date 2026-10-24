"""Explicit noncommercial Kaggle research authorization for Hala training.

No change to baseline/inference's commercial-use guard. Only training with
explicit user-designated noncommercial scope in a genuine private Kaggle
Linux notebook may enter via this narrowly scoped seam.
"""
import os
import platform
from pathlib import Path


def require_training_provider(config, *, env=None, system=None, workspace=None):
    env = os.environ if env is None else env
    system = platform.system() if system is None else system
    if system != "Linux" or env.get("HALA_REMOTE_TRAINING_ALLOWED") != "1":
        raise ValueError("explicitly approved Linux research training runtime required")
    if config.platform == "kaggle":
        if (env.get("HALA_NONCOMMERCIAL_RESEARCH") != "1"
                or not env.get("KAGGLE_KERNEL_RUN_TYPE")
                or env.get("COLAB_RELEASE_TAG")
                or not (Path("/kaggle/working").is_dir()
                        if workspace is None else workspace)):
            raise ValueError("noncommercial private Kaggle research runtime not verified")
        return "kaggle_noncommercial_research"
    if env.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle runtime requires explicit noncommercial research config")
    if config.platform == "colab":
        return "colab_research"
    raise ValueError("unapproved remote research provider")
