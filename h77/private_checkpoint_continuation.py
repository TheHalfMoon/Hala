"""H77: explicitly opt-in, fail-closed private checkpoint continuity after H65.

Host checkpoint staging is not private durable provider storage. This orchestration
never allocates GPUs and does not bypass the existing authenticated readback gate.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from h65.provider_publish import publish_latest
from h65.safe_diagnostic import supervise, write_receipt
from training.checkpoints import atomic_json, load_checkpoint

SCHEMA = "hala.h77.private-continuation.v1"


def supervise_then_private_readback(session, sha, destination, *,
                                   token_env="KAGGLE_API_TOKEN",
                                   monitor=supervise, publisher=publish_latest,
                                   environ=None, **monitor_kwargs):
    """Salvage a verified host checkpoint even when its T4 job failed.

    Never auto-delete the only host copy; do not alter FAILED monitor outcomes
    just because private readback of a partial step succeeded.
    """
    if (not isinstance(token_env, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", token_env)):
        raise ValueError("safe environment variable name required")
    env = os.environ if environ is None else environ
    token = env.get(token_env)
    if not isinstance(token, str) or len(token) < 10:
        raise ValueError("explicit private provider credential required before GPU probe")

    destination = Path(destination).resolve()
    original = monitor(session, sha, destination, **monitor_kwargs)
    if not isinstance(original, dict) or original.get("schema") != "hala.h65.supervised-diagnostic.v1":
        raise ValueError("H65 supervisor receipt must be structurally valid")
    result = {**original, "private_provider_readback_verified": False,
              "private_publication_status": "SKIPPED_NO_VERIFIED_HOST_CHECKPOINT"}
    receipt = {"schema": SCHEMA, "monitor_status": original.get("status"),
               "private_provider_readback_verified": False}
    if type(original.get("successful_egress_polls")) is int and original["successful_egress_polls"] > 0:
        try:
            staging = destination / "checkpoint-staging"
            _, state = load_checkpoint(staging, None)
            output = publisher(staging, token, destination / "private-publication",
                               remove_host_copy=False)
            if (not isinstance(output, dict)
                    or output.get("provider_published") is not True
                    or output.get("provider_is_private") is not True
                    or output.get("provider_readback_verified") is not True
                    or output.get("host_copy_removed") is not False
                    or output.get("step") != state["step"]
                    or not isinstance(output.get("private_provider_ref"), str)
                    or not output["private_provider_ref"]):
                raise ValueError("provider private readback or checkpoint identity NOT verified")
            # A mutable local directory is not a substitute for private readback.
            # Preserve it for independent recovery; never delete on success.
            load_checkpoint(staging, None)
            result.update(private_provider_readback_verified=True,
                          private_publication_status="PRIVATE_READBACK_VERIFIED",
                          private_provider_ref=output["private_provider_ref"],
                          private_checkpoint_step=state["step"])
            receipt.update(private_provider_readback_verified=True,
                           private_publication_status="PRIVATE_READBACK_VERIFIED",
                           private_checkpoint_step=state["step"])
        except Exception as error:
            result["private_publication_status"] = "PRIVATE_READBACK_FAILED"
            result["private_publication_error_class"] = type(error).__name__
            receipt.update(private_publication_status="PRIVATE_READBACK_FAILED",
                           private_publication_error_class=type(error).__name__)
            if result.get("status") != "FAILED":
                result["status"] = "FAILED"
                result["reason"] = "private_readback_failed"
    else:
        receipt["private_publication_status"] = "SKIPPED_NO_VERIFIED_HOST_CHECKPOINT"
        if result.get("status") != "FAILED":
            result["status"] = "FAILED"
            result["reason"] = "no_verified_host_checkpoint_for_private_readback"

    destination.mkdir(parents=True, exist_ok=True)
    # Separate result avoids changing H65's original immutable monitor verdict.
    atomic_json(destination / "h77-private-continuation.json", result)
    write_receipt(destination / "monitor-receipts.jsonl",
                  {"event": "h77_private_continuation", **receipt})
    return result
