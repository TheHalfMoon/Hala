"""H85: one session-local pinned Qwen cache across training and inference.

An explicit transformers cache_dir is needed: HF_HOME is often set after
transformers or huggingface_hub was imported, when defaults are cached.
The cache is disposable and shared only inside an approved remote session.
"""
from __future__ import annotations



from training.scripts.baseline import MODEL_ID, MODEL_REVISION

HF_HOME = "/tmp/hala-model-cache"
HUB_CACHE = HF_HOME + "/hub"


def pinned_cache_kwargs(model_id: str, revision: str) -> dict:
    """Validate exact pinned model source before resolving cache or downloads."""
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError("Only the immutable Hala Qwen3-VL model revision is qualified")
    return {"revision": revision, "trust_remote_code": False,
            "cache_dir": HUB_CACHE}
