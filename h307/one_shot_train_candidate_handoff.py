"""H307: one-shot, role-locked handoff of H286 TRAIN candidates.

This is NOT a dataset admission, trainer, or guarantee of downstream
transactional storage. The owner-provided consumer MUST itself be atomic:
a consumer that writes and then raises may leave partial external state.
Unlike H286's per-row sink, this boundary calls the external consumer
only ONCE, after all 24 source-browser candidates pass qualification.
Never use this for DEV, OOD, model scoring, or public media export.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Callable

from h285.train_only_browser_candidates import ROLE, SOURCE_BLOB
from h286.relative_candidate_training import produce_relative_train_candidates

COUNT = 24
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_BATCH_BYTES = COUNT * MAX_FRAME_BYTES
Batch = tuple[tuple[bytes, str, str], ...]


class CandidateHandoffRefused(ValueError):
    """Categorical error only: never include source prompts or labels."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise CandidateHandoffRefused("H307 " + reason)


def handoff_train_candidate_batch_once(
    browser: object,
    source_path: Path,
    trusted_atomic_consumer: Callable[[Batch], bool],
    *,
    role: str,
) -> dict:
    """Stage 24 H286 samples privately, then invoke one trusted batch sink.

    The sink MUST make its own all-or-nothing commit and return literal True.
    This boundary cannot roll back a sink's external side effects. A failure
    is terminal for this call; there is no retry or partial-batch callback.
    No screenshots, prompts, labels or per-case digests are returned/logged.
    """
    need(type(role) is str and role == ROLE, "TRAIN_CANDIDATE_ROLE_ONLY")
    need(isinstance(source_path, Path), "SOURCE_PATH_REQUIRED")
    need(callable(trusted_atomic_consumer), "ATOMIC_BATCH_CONSUMER_REQUIRED")
    staged: list[tuple[bytes, str, str]] = []
    seen: set[str] = set()
    total_bytes = 0

    def collect(png: bytes, prompt: str, completion: str) -> None:
        nonlocal total_bytes
        need(type(png) is bytes
             and 100 <= len(png) < MAX_FRAME_BYTES
             and png.startswith(b"\x89PNG\r\n\x1a\n")
             and type(prompt) is str and 0 < len(prompt) <= 2000
             and type(completion) is str and 0 < len(completion) <= 1200,
             "INVALID_PRIVATE_CANDIDATE_SHAPE")
        need(len(staged) < COUNT, "EXCESS_CANDIDATES")
        total_bytes += len(png)
        need(total_bytes <= MAX_BATCH_BYTES, "BATCH_MEMORY_LIMIT")
        digest = sha256(png).hexdigest()
        need(digest not in seen, "DUPLICATE_CANDIDATE_IMAGE")
        seen.add(digest)
        staged.append((png, prompt, completion))

    try:
        proof = produce_relative_train_candidates(
            browser, source_path, collect, role=role)
    except Exception:
        # No external consumer has been invoked. Never echo raw source data.
        raise CandidateHandoffRefused("H307 SOURCE_BATCH_FAILED") from None

    need(type(proof) is dict
         and proof.get("status") ==
             "H286_RELATIVE_ALIGNED_FIRSTPARTY_TRAIN_CANDIDATES_ONLY"
         and proof.get("original_source_blob") == SOURCE_BLOB
         and proof.get("role") == ROLE
         and type(proof.get("actual_browser_supervisions")) is int
         and proof["actual_browser_supervisions"] == COUNT
         and type(proof.get("relative_targets_roundtrip_verified")) is int
         and proof["relative_targets_roundtrip_verified"] == COUNT
         and type(proof.get("english_candidates")) is int
         and proof["english_candidates"] == 12
         and type(proof.get("arabic_candidates")) is int
         and proof["arabic_candidates"] == 12
         and type(proof.get("real_model_generations")) is int
         and proof["real_model_generations"] == 0
         and type(proof.get("optimizer_steps")) is int
         and proof["optimizer_steps"] == 0
         and proof.get("official_train_admitted") is False
         and proof.get("source_disjoint_blind_ood") is False
         and proof.get("adapted_gui_gain_proven") is False
         and proof.get("media_or_labels_saved_to_public_output") is False
         and len(staged) == COUNT and len(seen) == COUNT,
         "UNQUALIFIED_OR_INCOMPLETE_TRAIN_CANDIDATES")

    # Exactly one external handoff, never 24 independent partial writes.
    # The callback owns transactionality; a partial external commit is NOT
    # something this process can detect, undo, or certify as absent.
    try:
        committed = trusted_atomic_consumer(tuple(staged))
    except Exception:
        raise CandidateHandoffRefused(
            "H307 CONSUMER_FAILED_EXTERNAL_ROLLBACK_UNVERIFIED") from None
    need(committed is True, "CONSUMER_DID_NOT_ATTEST_ATOMIC_COMMIT")
    return {
        "status": "H307_ONE_SHOT_TRAIN_CANDIDATE_HANDOFF_ONLY",
        "source_role": ROLE,
        "candidate_count": COUNT,
        "external_consumer_calls": 1,
        "consumer_self_attested_atomic_commit": True,
        "external_storage_atomicity_independently_proven": False,
        "official_train_admitted": False,
        "blind_ood_evaluated": False,
        "model_generations": 0,
        "optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "can_release": False,
    }
