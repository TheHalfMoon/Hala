"""Freeze authored data, settings, source and real qualified media before inference."""
import hashlib
import json
from pathlib import Path
from development.data import build_dataset, canonical, validate_dataset
from development.media import render_frame
from development.simulator import Simulator, oracle
from development.verifier import verify_final


def freeze_bundle(output, settings, font_path=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to replace an existing freeze')
    if not isinstance(settings, dict) or not settings:
        raise ValueError('explicit runtime settings required')
    output.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    validation = validate_dataset(data)
    images, materialized = {}, []
    if font_path is not None:
        episodes = {e['id']: e for e in data['episodes']}
        states = [(r['episode'], r['state']) for r in data['rows']]
        for episode in data['episodes']:
            if episode['split'] == 'train':
                continue
            sim = Simulator(episode)
            while not verify_final(episode, sim.snapshot()):
                states.append((episode['id'], sim.snapshot()))
                sim.apply(oracle(sim))
        for episode_id, state in states:
            key = hashlib.sha256(canonical({'episode': episode_id, 'state': state})).hexdigest()
            if key not in images:
                relative = f'images/{key}.png'
                images[key] = dict(render_frame(episodes[episode_id], state, output / relative, font_path), path=relative)
        for row in data['rows']:
            row = dict(row)
            key = hashlib.sha256(canonical({'episode': row['episode'], 'state': row['state']})).hexdigest()
            row.update(image=images[key]['path'], image_sha256=images[key]['sha256'])
            materialized.append(row)
    files = {'dataset.json': canonical(data), 'settings.json': canonical(settings),
             'media.json': canonical(images), 'training-rows.jsonl': b''.join(canonical(r)+b'\n' for r in materialized)}
    source = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}
    for name, content in files.items():
        (output / name).write_bytes(content)
    manifest = {'schema': 'hala.development.freeze.v1', 'validation': validation,
                'files': {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
                'source': source_hashes, 'media_qualified': bool(images), 'image_count': len(images),
                'training_rows': len(materialized), 'native_qualified': False}
    (output / 'manifest.json').write_bytes(canonical(manifest))
    return manifest


def verify_bundle(path, require_media=True, expected_manifest_sha256=None):
    path = Path(path)
    actual_manifest_hash = hashlib.sha256((path / 'manifest.json').read_bytes()).hexdigest()
    if require_media and expected_manifest_sha256 is None:
        raise ValueError('independently archived manifest SHA256 required')
    if expected_manifest_sha256 is not None and actual_manifest_hash != expected_manifest_sha256:
        raise ValueError('external manifest hash mismatch')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['files']) != {'dataset.json', 'settings.json', 'media.json', 'training-rows.jsonl'}:
        raise ValueError('unexpected freeze files')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError('freeze file hash mismatch')
    validate_dataset(json.loads((path / 'dataset.json').read_text(encoding='utf-8')))
    source = Path(__file__).parent
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}
    if actual != manifest['source']:
        raise ValueError('frozen source differs from loaded implementation')
    images = json.loads((path / 'media.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (path / 'training-rows.jsonl').read_text(encoding='utf-8').splitlines()]
    if (type(manifest['media_qualified']) is not bool or manifest['media_qualified'] != bool(images) or
            manifest['image_count'] != len(images) or manifest['training_rows'] != len(rows)):
        raise ValueError('inconsistent qualification claims')
    if require_media and not manifest['media_qualified']:
        raise ValueError('actual media qualification missing')
    if images:
        expected_rows = []
        for row in build_dataset()['rows']:
            key = hashlib.sha256(canonical({'episode': row['episode'], 'state': row['state']})).hexdigest()
            if key not in images:
                raise ValueError('missing training image')
            expected_rows.append(dict(row, image=images[key]['path'], image_sha256=images[key]['sha256']))
        if canonical(rows) != canonical(expected_rows):
            raise ValueError('materialized row drift')
    elif rows:
        raise ValueError('rows without qualified media')
    for key, record in images.items():
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key) or record['path'] != f'images/{key}.png':
            raise ValueError('noncanonical media path')
        from development.media import FONT_SHA256
        if (record['font_sha256'] != FONT_SHA256 or record['pillow'] != '11.3.0' or record['raqm'] != '0.10.1' or
                record['platform'] != 'Linux' or (record['width'], record['height']) != (800, 600)):
            raise ValueError('unqualified media metadata')
        if hashlib.sha256((path / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('image hash mismatch')
    return manifest
