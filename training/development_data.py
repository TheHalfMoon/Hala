"""Reviewed semantic Mind/Type training projection; historical action loader unchanged."""
import hashlib
import json
from io import BytesIO
from pathlib import Path

from development.freeze import verify_bundle
from evaluation.development_runtime import check_settings
from training.artifacts import artifact_path, strict_json
from training.masking import completion_labels

# Independently bound after freeze verification; no unchecked second read
# may be passed to the processor or historical development trainer.
FROZEN_PROJECTION_BYTE_LIMIT = 16 * 1024 * 1024




def _verified_frozen_manifest(root, manifest_sha256):
    """Pin training projection to original external manifest bytes.

    The historical freeze verifier is byte-immutable by contract and
    cannot be edited. It reopens manifest.json internally; compare its
    returned metadata to this independently hash-bound snapshot instead.
    """
    if (type(manifest_sha256) is not str or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)):
        raise ValueError("Frozen manifest needs an exact external SHA-256")
    path = root / "manifest.json"
    if path.is_symlink():
        raise ValueError("Symlinked frozen manifest forbidden")
    with path.open("rb") as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if (not raw or len(raw) > 4 * 1024 * 1024
            or hashlib.sha256(raw).hexdigest() != manifest_sha256):
        raise ValueError("Frozen manifest differs from independently pinned SHA")
    snapshot = strict_json(raw.decode("utf-8"))
    if type(snapshot) is not dict:
        raise ValueError("Frozen manifest must be a JSON object")
    checked = verify_bundle(root, require_media=True,
                            expected_manifest_sha256=manifest_sha256)
    if (json.dumps(checked, sort_keys=True, allow_nan=False)
            != json.dumps(snapshot, sort_keys=True, allow_nan=False)):
        raise ValueError("Frozen manifest changed after external SHA check")
    return snapshot

def _verified_projection_bytes(root, manifest, name):
    """Single bounded snapshot whose exact bytes match the verified freeze.

    verify_bundle checks the frozen source/rows before we get here. These
    SAME bytes are then decoded and projected; mutations after verification
    cannot substitute unverified Mind/Type prompts or targets.
    """
    if name not in ("settings.json", "training-rows.jsonl"):
        raise ValueError("Unregistered freeze projection file")
    expected = manifest["files"][name]
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Invalid freeze projection digest")
    path = root / name
    if path.is_symlink():
        raise ValueError("Symlinked freeze projection forbidden")
    with path.open("rb") as handle:
        raw = handle.read(FROZEN_PROJECTION_BYTE_LIMIT + 1)
    if (not raw or len(raw) > FROZEN_PROJECTION_BYTE_LIMIT
            or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError("Frozen projection bytes differ from verified manifest")
    return raw


def load_rows(bundle, manifest_sha256):
    root = Path(bundle).resolve()
    manifest = _verified_frozen_manifest(root, manifest_sha256)
    settings_raw = _verified_projection_bytes(root, manifest, "settings.json")
    rows_raw = _verified_projection_bytes(root, manifest, "training-rows.jsonl")
    check_settings(strict_json(settings_raw.decode("utf-8")))
    # verify_bundle independently reconstructs ALL authored rows and deterministic
    # labels, including family allocation and media identities, before projection.
    # Project from the exact hash-bound snapshot, never a later unverified read.
    materialized = [strict_json(line.decode("utf-8"))
                    for line in rows_raw.splitlines()]
    selected = []
    for row in materialized:
        if row['component'] == 'reliance':
            continue
        if row['split'] != 'train' or row['component'] not in ('mind', 'type'):
            raise ValueError('Only authored train Mind/Type rows allowed')
        selected.append({key: row[key] for key in ('id', 'family', 'language', 'component', 'prompt', 'completion')} |
                        {'image': row['image'] if row['component'] == 'type' else None,
                         'image_sha256': row['image_sha256'] if row['component'] == 'type' else None})
    if not selected or {r['language'] for r in selected} != {'en', 'ar'}:
        raise ValueError('Bilingual training rows required')
    return selected, manifest


def encode_row(row, processor, root, image_open, token_budget=4096):
    """Only public prompt, completion and current Type PNG reach the processor."""
    image = None
    if (row['component'] == 'type') != (row['image'] is not None):
        raise ValueError('Mind text only; Type current image required')
    if row['image'] is not None:
        path = artifact_path(Path(root).resolve(), row['image'])
        size = path.stat().st_size
        if not 8 <= size <= 2 * 1024 * 1024:
            raise ValueError('Current PNG byte budget mismatch')
        with path.open('rb') as handle:
            raw = handle.read(2 * 1024 * 1024 + 1)
        if (not 8 <= len(raw) <= 2 * 1024 * 1024 or not raw.startswith(b'\x89PNG\r\n\x1a\n') or
                len(raw) != size or hashlib.sha256(raw).hexdigest() != row['image_sha256']):
            raise ValueError('Current PNG digest/byte budget mismatch')
        # H349: decode the exact bytes we just hash-verified, not a second
        # mutable-path read that could substitute unapproved model pixels.
        with image_open(BytesIO(raw)) as source:
            if source.format != 'PNG' or source.size != (800, 600):
                raise ValueError('Qualified PNG dimensions required')
            image = source.convert('RGB')
    try:
        content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': row['prompt']}]
        messages = [{'role': 'user', 'content': content}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': row['completion']}],
                                             tokenize=False, add_generation_prompt=False)
        kwargs = {'return_tensors': 'pt'}
        if image is not None:
            kwargs['images'] = [image]
        inputs = processor(text=[full], **kwargs)
        prompt = processor(text=[prefix], **kwargs)
        if inputs.input_ids.shape[0] != 1 or prompt.input_ids.shape[0] != 1:
            raise ValueError('Batch one required')
        labels = completion_labels(inputs.input_ids[0].tolist(), prompt.input_ids[0].tolist(), token_budget)
        return inputs, labels
    finally:
        if image is not None:
            image.close()
