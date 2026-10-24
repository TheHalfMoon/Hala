"""Noncommercial Hala Kaggle research gate, separate from commercial releases.

Kaggle container images may also set COLAB_RELEASE_TAG; Kaggle's actual
provider marker and its /kaggle/working workspace are authoritative.
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
                or not (Path("/kaggle/working").is_dir()
                        if workspace is None else workspace)):
            raise ValueError("noncommercial private Kaggle research runtime not verified")
        return "kaggle_noncommercial_research"
    if env.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle runtime requires explicit noncommercial research config")
    if config.platform == "colab":
        return "colab_research"
    raise ValueError("unapproved remote research provider")
