"""Bounded, content-bound SFT inputs. No model or GPU imports occur here."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from data.factory import RIGHTS

MAX_PIXELS = 16000000
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_RECORDS = 10000
ROW_FIELDS = {'id', 'split', 'family', 'language', 'prompt', 'completion',
              'image_path', 'image_sha256'}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value

    def constant(value):
        raise ValueError('Nonfinite JSON constant')

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (TypeError, RecursionError) as error:
        raise ValueError('Invalid bounded JSON') from error


def normalized_relative_path(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 240 or
            '\\' in value or ':' in value or '\x00' in value):
        raise ValueError('Normalized relative artifact path required')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('Normalized relative artifact path required')
    return path


def artifact_path(root, value):
    relative = normalized_relative_path(value)
    path = root.joinpath(*relative.parts)
    # Reject symlink components as well as traversal; identities bind local bytes.
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError('Symlink artifact is forbidden')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes dataset')
    return path


def validate_action_completion(text):
    value = strict_json(text)
    fields = {'semantic_invoke': {'type', 'element_ref'},
              'click': {'type', 'point', 'coordinate_space', 'image_scale', 'element_ref_hint'},
              'set_value': {'type', 'element_ref', 'text', 'secret'},
              'scroll': {'type', 'element_ref', 'delta_x', 'delta_y'},
              'observe': {'type'}}
    if (not isinstance(value, dict) or not isinstance(value.get('type'), str) or
            set(value) != fields.get(value['type'])):
        raise ValueError('Exact logical action completion required')
    for key in ('element_ref', 'element_ref_hint'):
        if key in value and (not isinstance(value[key], str) or not 1 <= len(value[key]) <= 256):
            raise ValueError('Bounded action target required')
    if value['type'] == 'set_value':
        if (not isinstance(value['text'], str) or len(value['text']) > 2048 or
                value['secret'] is not False):
            raise ValueError('Bounded nonsecret synthetic input required')
    if value['type'] == 'scroll':
        if any(type(value[key]) is not int or abs(value[key]) > 10000
               for key in ('delta_x', 'delta_y')):
            raise ValueError('Bounded scroll delta required')
    if value['type'] == 'click':
        point = value['point']
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) or
                    not 0 <= x <= 1000000 for x in point) or
                value['coordinate_space'] != 'window' or
                type(value['image_scale']) not in (int, float) or
                value['image_scale'] not in (1, 1.5, 2)):
            raise ValueError('Bounded synthetic click geometry required')
    return value


def check_image_header(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            if image.format != 'PNG' or image.width * image.height > MAX_PIXELS:
                raise ValueError('Bounded PNG image required before conversion')
            image.verify()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Invalid bounded PNG image') from error


def load_sft_artifact(examples_path, manifest):
    """Verify JSONL and every media byte before any processor/model allocation."""
    examples = Path(examples_path)
    root = examples.parent.resolve()
    # Never launder first-party Edge captures into the legacy Pillow fixture
    # provenance. Admit only the exact, independently reconstructed H48 export.
    from h48.factory import PROVENANCE as H48_RIGHTS, SCHEMA as H48_SCHEMA
    h48_mode = manifest.get('source_schema') == H48_SCHEMA
    if h48_mode:
        from h48.sft_bridge import expected_bytes
        _, _, expected_manifest = expected_bytes(root)
        if manifest != expected_manifest:
            raise ValueError('H48 source, rights, split, render route or labels changed')
    if (manifest.get('schema') != 'hala.training.manifest.v1' or
            manifest.get('split') != 'train' or manifest.get('source_kind') != 'synthetic' or
            manifest.get('rights_cleared') is not True):
        raise ValueError('Original synthetic train-only artifact required')
    if examples.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError('Dataset exceeds byte budget')
    budget = manifest.get('token_budget')
    if type(budget) is not int or not 128 <= budget <= 8192:
        raise ValueError('Declared bounded token budget required')
    families, held_out = manifest.get('families'), manifest.get('held_out_families')
    if (not isinstance(families, list) or not families or
            not isinstance(held_out, list) or not held_out or
            any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in families + held_out) or
            len(set(families)) != len(families) or len(set(held_out)) != len(held_out) or
            set(families) & set(held_out)):
        raise ValueError('Separate bounded family identities required')
    arabic = manifest.get('arabic_image_training')
    if (not isinstance(arabic, dict) or
            type(arabic.get('enabled')) is not bool or
            type(arabic.get('raqm_qualified')) is not bool or
            (arabic['enabled'] and not (arabic['raqm_qualified'] or h48_mode))):
        raise ValueError('Explicit Arabic shaping eligibility required')
    media_list = manifest.get('media')
    if not isinstance(media_list, list) or not 1 <= len(media_list) <= MAX_RECORDS:
        raise ValueError('Bounded reviewed media manifest required')
    media = {}
    for item in media_list:
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256', 'family', 'provenance'} or
                item['provenance'] != (H48_RIGHTS if h48_mode else RIGHTS) or item['family'] not in families or
                not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            raise ValueError('Content-bound original synthetic media rights required')
        normalized_relative_path(item['path'])
        if item['path'] in media:
            raise ValueError('Duplicate media identity')
        media[item['path']] = item
    records, ids, used_media, used_families = [], set(), set(), set()
    jsonl_digest, bytes_read = hashlib.sha256(), 0
    with examples.open('rb') as handle:
        for raw_line in handle:
            bytes_read += len(raw_line)
            jsonl_digest.update(raw_line)
            if bytes_read > MAX_DATASET_BYTES or len(raw_line) > 200000 or not raw_line.strip():
                raise ValueError('Bounded nonempty JSONL row required')
            row = strict_json(raw_line.decode('utf-8'))
            if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                raise ValueError('Exact SFT row fields required')
            if (not isinstance(row['id'], str) or not 1 <= len(row['id']) <= 256 or
                    row['id'] in ids or row['split'] != 'train' or row['family'] not in families or
                    row['language'] not in ('en', 'ar') or
                    (row['language'] == 'ar' and not arabic['enabled'])):
                raise ValueError('Unique original train-only record identity required')
            if (not isinstance(row['prompt'], str) or not 1 <= len(row['prompt']) <= 32768 or
                    not isinstance(row['completion'], str) or not 1 <= len(row['completion']) <= 4096):
                raise ValueError('Bounded nonempty prompt/completion required')
            validate_action_completion(row['completion'])
            path = artifact_path(root, row['image_path'])
            item = media.get(row['image_path'])
            if (item is None or item['family'] != row['family'] or
                    row['image_sha256'] != item['sha256']):
                raise ValueError('SFT row media digest/family binding mismatch')
            if path.stat().st_size > MAX_IMAGE_BYTES or sha256_file(path) != item['sha256']:
                raise ValueError('Training media digest/byte budget mismatch')
            check_image_header(path)
            ids.add(row['id'])
            used_media.add(row['image_path'])
            used_families.add(row['family'])
            records.append(row)
            if len(records) > MAX_RECORDS:
                raise ValueError('Training record budget exceeded')
    if jsonl_digest.hexdigest() != manifest.get('dataset_sha256'):
        raise ValueError('Training JSONL digest mismatch')
    if not records or used_media != set(media) or used_families != set(families):
        raise ValueError('Manifest must bind exactly the used nonempty media and families')
    return records
