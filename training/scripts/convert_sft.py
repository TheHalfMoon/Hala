"""Convert reviewed ORIGINAL synthetic train records into image/action SFT inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from data.factory import manifest_bytes, training_records
from evaluation.engine import Environment, OracleProvider
from evaluation.render import render_case
from training.artifacts import load_sft_artifact, sha256_file, strict_json


def convert(source_manifest, destination, font_path=None, raqm_qualified=False):
    # Importer validates complete original task identities and refuses relabeled holdouts.
    cases = training_records(source_manifest)
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refusing to overwrite SFT artifact')
    from PIL import features
    actual_raqm = features.check_feature('raqm')
    if type(raqm_qualified) is not bool or (raqm_qualified and not actual_raqm):
        raise ValueError('Arabic image training requires actual qualified RAQM shaping')
    destination.mkdir(parents=True)
    rows, media = [], []
    excluded_arabic = 0
    for case in cases:
        if case['language'] == 'ar' and not raqm_qualified:
            excluded_arabic += 1
            continue
        image_path = 'images/' + case['task_id'] + '.png'
        render_case(case, destination / image_path, font_path)
        environment = Environment(case)
        request = environment.observe()
        action = OracleProvider(case).propose(request)['actions'][0]
        # Supervise only the action; provider metadata and privileged final state are absent.
        observed = request.observation.to_dict()['accessibility']
        prompt = ('Propose one logical action for this original synthetic GUI fixture.\n'
                  'Goal: ' + case['goal'] + '\nObserved controls: ' +
                  json.dumps(observed, ensure_ascii=False, sort_keys=True) + '\n'
                  'Return only the action JSON object. Local policy and independent state verification remain authoritative.')
        digest = sha256_file(destination / image_path)
        rows.append({'id': case['task_id'], 'split': 'train', 'family': case['family_id'],
                     'language': case['language'], 'prompt': prompt,
                     'completion': json.dumps(action, ensure_ascii=False, sort_keys=True),
                     'image_path': image_path, 'image_sha256': digest})
        media.append({'path': image_path, 'sha256': digest, 'family': case['family_id'],
                      'provenance': case['provenance']})
    examples = destination / 'examples.jsonl'
    examples.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    bound = {'schema': 'hala.training.manifest.v1', 'split': 'train',
             'rights_cleared': True, 'source_kind': 'synthetic',
             'source_schema': source_manifest['schema'],
             'source_sha256': hashlib.sha256(manifest_bytes(source_manifest)).hexdigest(),
             'dataset_sha256': sha256_file(examples),
             'families': sorted({row['family'] for row in rows}),
             'held_out_families': sorted({case['family_id'] for case in source_manifest['tasks']
                                          if case['split'] != 'train'}),
             'token_budget': 4096, 'media': media,
             'label_scope': 'First oracle logical action from initial synthetic observation only',
             'arabic_image_training': {'enabled': raqm_qualified,
                                       'raqm_available': actual_raqm,
                                       'raqm_qualified': raqm_qualified,
                                       'excluded_records': excluded_arabic,
                                       'reason': 'Arabic images excluded until actual shaping is independently qualified'},
             'training_executed': False, 'adapter_qualified': False}
    manifest = destination / 'manifest.json'
    manifest.write_text(json.dumps(bound, indent=2, ensure_ascii=False,
                                   sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    load_sft_artifact(examples, bound)
    return examples, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font')
    parser.add_argument('--raqm-qualified', action='store_true')
    args = parser.parse_args()
    source = strict_json(Path(args.source).read_text(encoding='utf-8'))
    examples, manifest = convert(source, args.output, args.font, args.raqm_qualified)
    print(json.dumps({'examples': str(examples), 'manifest': str(manifest),
                      'training_executed': False}))


if __name__ == '__main__':
    main()
