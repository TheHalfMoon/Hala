"""Reviewed semantic Mind/Type training projection; historical action loader unchanged."""
import hashlib
import json
from pathlib import Path

from development.freeze import verify_bundle
from evaluation.development_runtime import check_settings
from training.artifacts import artifact_path
from training.masking import completion_labels


def load_rows(bundle, manifest_sha256):
    root = Path(bundle).resolve()
    manifest = verify_bundle(root, require_media=True, expected_manifest_sha256=manifest_sha256)
    check_settings(json.loads((root / 'settings.json').read_text(encoding='utf-8')))
    # verify_bundle independently reconstructs ALL authored rows and deterministic
    # labels, including family allocation and media identities, before projection.
    materialized = [json.loads(line) for line in (root / 'training-rows.jsonl').read_text(encoding='utf-8').splitlines()]
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
        with image_open(path) as source:
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
