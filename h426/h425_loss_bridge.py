"""H426/H425 actual source-loss comparison seam (RESEARCH, never trains).

This module evaluates H425's ORIGINAL frozen numeric-token objective and
H426's source-target-aware objective on the SAME already-created image-
conditioned output/labels. It has no optimizer, model or GPU launch,
source right admission, public release, or new accuracy claim.

The caller must produce labels from h425/.../to_inputs, which alone handles
image decoding, verified source hashes, the visual Qwen processor and
the assistant-only supervision boundary. An independent, signed rights gate
is still required before any NEW data-dependent training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H425_SOURCE = ROOT/"h425/kaggle/h425_h417_replay_paired.py"
H426_SOURCE = ROOT/"h426/target_aware_loss.py"
DOMAINS = frozenset(("slider", "slide_resize", "text_highlight", "sheet"))
LANGUAGES = frozenset(("en", "ar"))


def source_case_identity(case):
    """Fail closed on malformed H425 research microbatch identity.

    This only establishes the expected data SHAPE, not permission to TRAIN.
    Source ZIP/JPEG/label/DEV exclusion is the frozen H425 loader's job.
    """
    if type(case) is not dict:
        raise ValueError("H426 H425 source case must be an original dictionary")
    if (type(case.get("domain")) is not str
        or case["domain"] not in DOMAINS
        or type(case.get("language")) is not str
        or case["language"] not in LANGUAGES
        or type(case.get("id")) is not str
        or not 1 <= len(case["id"]) <= 512
        or type(case.get("image_sha256")) is not str
        or len(case["image_sha256"]) != 64
        or any(x not in "0123456789abcdef" for x in case["image_sha256"])
        or type(case.get("body")) is not str
        or not 1 <= len(case["body"]) <= 4096):
        raise ValueError("H426 H425 image/domain/language/source identity invalid")
    return {
        "domain": case["domain"],
        "language": case["language"],
        "image_sha256": case["image_sha256"],
        "id_sha256": hashlib.sha256(case["id"].encode("utf8")).hexdigest(),
    }


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("H426 original H425/H426 training source unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_original_h425_to_h426(output, labels, processor, case):
    """Return two live differentiable Torch loss scalars for one native drag.

    The H425 baseline function is imported verbatim from the original
    generated H425 source, not an approximate reimplementation. No
    backward(), optimizer step, dataset read, checkpoint save or upload.
    """
    row = source_case_identity(case)
    import torch
    if (not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.ndim != 2
        or labels.shape[0] != 1
        or labels.shape[1] < 8):
        raise ValueError("H426 H425 labels must be a single integral sequence")
    if not isinstance(getattr(output, "logits", None), torch.Tensor):
        raise ValueError("H426 H425 source model output logits missing")
    logits = output.logits
    if (logits.ndim != 3 or logits.shape[0] != 1
        or logits.shape[1] != labels.shape[1]
        or logits.shape[2] < 32):
        raise ValueError("H426 H425 original forward/label tensor alignment drift")
    keep = labels[:, 1:] != -100
    if not bool(keep.any().item()):
        raise ValueError("H426 H425 original assistant labels absent")
    # Assistant-only labels must be an EXACT continuous final suffix.
    original_indices = torch.nonzero(keep[0], as_tuple=False).flatten().tolist()
    if original_indices != list(range(original_indices[0], keep.shape[1])):
        raise ValueError("H426 H425 assistant supervision suffix fragmented")
    if (not bool(torch.isfinite(logits).all().item())
        or int(torch.min(labels[keep.new_ones(labels.shape, dtype=torch.bool)]).item()) < -100):
        raise ValueError("H426 H425 source tensors nonfinite or malformed labels")
    h425 = _load_source_module("h426_actual_h425_source", H425_SOURCE)
    h426 = _load_source_module("h426_target_loss_source", H426_SOURCE)
    old = h425.coordinate_weighted_loss(output, labels, processor)
    new = h426.proposed_h426_weighted_ce(output, labels, processor, row["domain"])
    if not (bool(torch.isfinite(old).item())
            and bool(torch.isfinite(new).item())
            and old.requires_grad and new.requires_grad):
        raise ValueError("H426 H425 paired loss evidence nonfinite/no gradient")
    return {
        "h425_original_loss": old,
        "h426_target_loss": new,
        "domain": row["domain"],
        "source_identity_sha": row["id_sha256"],
        "no_training_performed": True,
    }
