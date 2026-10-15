"""H18 deterministic inventory and fail-closed prospective adaptation policy.

No training entrypoint, generic data importer, model imports or job submission.
Only the existing original synthetic candidate is admitted for inventory.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from data.factory import load_frozen, manifest_bytes, training_records
from evaluation.engine import Environment, OracleProvider
from evaluation.heldout_protocol import score
from training.artifacts import load_sft_artifact, sha256_file, strict_json


# Historical evidence identities reviewed before H18. H15's independent record
# binds the candidate examples/manifest and its engineering report. H14's measured
# raw result and prior freeze are separately pinned, not authenticated by one another.
# This is deliberately a fixed-snapshot inventory: new data needs a new review.
HISTORICAL_SHA256 = {
    'data/frozen/tasks.json': 'bdde4ea5326b391172435e29cee77701135c8bfa73805856ffa5dab538bbd35e',
    'training/generated-candidate/examples.jsonl': '60c79bf7d06f6fc436e92bbfd35a86cb3bbf3b751df9fb13246cc74410a30c91',
    'training/generated-candidate/manifest.json': '95dfe1614bcc4a2dcd9942431350958005f6d07abbf14efa10a38d3d13ee3aa6',
    'evaluation/heldout/tasks.json': '0efd21686fb1fc6255ffe463256f290c8d1cbb2fbcd61ad8d982ae1c5990e6c5',
    'evaluation/heldout/manifest.json': '0d81cc61ecaafb79b52b6a0368f76d550c92f5e4c5ccb05ef468e92a83c5b8d6',
    'experiments/heldout/h14-fp16-result.json': '6a1adf9a4dd322c06e07ed317f1d880763c60c214d1cc87c036dea21567a7a32',
    'experiments/runtime/h15-lora-qualification.json': '2d638e61cc1d8c536ec7793204b249a9d7bdfe1807f4af71b07eee4ed5918257',
    'experiments/runtime/h15-independent-checkpoint-verification.json': '0370da9427b6e14df7ec34c31c01ee4d76d6a08f085b37bdd502bdf640c9cf1c',
}


def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def counts(rows, key):
    return dict(sorted(Counter(row[key] for row in rows).items()))


def validate_config(config):
    """Accept this preparation spec only; no Boolean can enable an unbuilt runner."""
    expected = {
        'schema': 'hala.h18.prospective-shared-lora.v1',
        'status': 'blocked_preparation_only',
        'base_model': 'Qwen/Qwen3-VL-2B-Instruct',
        'base_revision': '89644892e4d85e24eaac8bacfd4f463576704203',
        'architecture': 'single_shared_base_single_multitask_lora',
        'adapter_count': 1,
        'tasks': ['mind_plan_v1', 'type_normalized_point_v1'],
        'languages': ['en', 'ar'],
        'datasets': [],
        'training_enabled': False,
        'download_weights': False,
        'submit_job': False,
        'candidate_hyperparameters': {'precision': 'fp16', 'rank': 4,
            'learning_rate': 0.00005, 'batch_size': 1, 'gradient_accumulation': 1},
        'data_requirements': ['rights_cleared_content_bound', 'validated_mind_plans',
            'normalized_type_points', 'bilingual_semantic_coverage',
            'qualified_arabic_rendering', 'independent_family_disjoint_train_dev_test',
            'h14_never_training_or_prompt_tuning'],
    }
    # Canonical JSON also distinguishes true from 1 and forbids nonfinite values.
    try:
        same = json.dumps(config, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid prospective config') from error
    if not same:
        raise ValueError('Only the disabled H18 prospective shared-model specification is supported')


def audit(root, config=None):
    """Recompute evidence from local bytes; malformed evidence raises ValueError.

    A successful audit means inventory integrity, never data/training readiness.
    Missing future data always leaves hard stops; caller claims cannot override it.
    """
    root = Path(root)
    if config is not None:
        validate_config(config)
    for relative, expected in HISTORICAL_SHA256.items():
        if sha256_file(root / relative) != expected:
            raise ValueError('Reviewed historical evidence changed: ' + relative)
    source = load_frozen(root / 'data/frozen/tasks.json')
    source_train = training_records(source)
    base = root / 'training/generated-candidate'
    manifest = read(base / 'manifest.json')
    arabic = manifest['arabic_image_training']
    excluded_arabic = sum(case['language'] == 'ar' for case in source_train)
    if (manifest['label_scope'] != 'First oracle logical action from initial synthetic observation only' or
            any(arabic[key] is not False for key in ('enabled', 'raqm_available', 'raqm_qualified')) or
            type(arabic['excluded_records']) is not int or arabic['excluded_records'] != excluded_arabic):
        raise ValueError('Historical first-action/Arabic exclusion evidence required')
    rows = load_sft_artifact(base / 'examples.jsonl', manifest)
    source_hash = hashlib.sha256(manifest_bytes(source)).hexdigest()
    if manifest.get('source_sha256') != source_hash:
        raise ValueError('SFT source digest does not bind frozen original tasks')
    eligible = {case['task_id']: case for case in source_train
                if case['language'] == 'en' or manifest['arabic_image_training']['enabled']}
    if set(eligible) != {row['id'] for row in rows}:
        raise ValueError('Candidate must contain exactly eligible original source identities')
    for row in rows:
        case = eligible[row['id']]
        request = Environment(case).observe()
        observed = request.observation.to_dict()['accessibility']
        expected_prompt = ('Propose one logical action for this original synthetic GUI fixture.\n'
            'Goal: ' + case['goal'] + '\nObserved controls: ' +
            json.dumps(observed, ensure_ascii=False, sort_keys=True) + '\n'
            'Return only the action JSON object. Local policy and independent state verification remain authoritative.')
        action = OracleProvider(case).propose(request)['actions'][0]
        if (row['image_path'] != 'images/' + case['task_id'] + '.png' or
                row['prompt'] != expected_prompt or strict_json(row['completion']) != action or
                row['family'] != case['family_id'] or row['language'] != case['language']):
            raise ValueError('Candidate differs from original prompt/first-action/source binding')

    heldout = read(root / 'evaluation/heldout/tasks.json')
    freeze = read(root / 'evaluation/heldout/manifest.json')
    if (heldout.get('training_allowed') is not False or
            sha256_file(root / 'evaluation/heldout/tasks.json') !=
            freeze['files']['evaluation/heldout/tasks.json']):
        raise ValueError('H14 training exclusion/frozen task digest mismatch')
    h14_cases = {case['id']: case for case in heldout['cases']}
    if (len(h14_cases) != 64 or set(h14_cases) & set(eligible) or
            {case['family'] for case in heldout['cases']} & {row['family'] for row in rows}):
        raise ValueError('H14 identities/families must be separate from training')
    result = read(root / 'experiments/heldout/h14-fp16-result.json')
    measured = result['cases']
    if len(measured) != 64 or {row['id'] for row in measured} != set(h14_cases):
        raise ValueError('H14 result must cover each frozen case once')
    rescored = []
    for row in measured:
        case = h14_cases[row['id']]
        current = score(case, row['output'])
        if current['pixel_point'] is not None:
            current['pixel_point'] = list(current['pixel_point'])
        if (row['language'] != case['language'] or row['family'] != case['family'] or
                any(row[key] != current[key] for key in current)):
            raise ValueError('H14 recorded score differs from frozen-case recomputation')
        rescored.append({**current, 'language': case['language'], 'family': case['family']})

    h15_path = root / 'experiments/runtime/h15-lora-qualification.json'
    h15 = read(h15_path)
    independent = read(root / 'experiments/runtime/h15-independent-checkpoint-verification.json')
    if sha256_file(h15_path) != independent['report_sha256']:
        raise ValueError('H15 report differs from independent verification identity')
    if (independent['exact'] is not True or h15['adapter_quality_qualified'] is not False or
            h15['arabic_remediation_claimed'] is not False or
            independent['inputs']['examples'] != HISTORICAL_SHA256['training/generated-candidate/examples.jsonl'] or
            independent['inputs']['manifest'] != HISTORICAL_SHA256['training/generated-candidate/manifest.json']):
        raise ValueError('Exact limited H15 engineering evidence required; no quality claims')

    def summary(items):
        return {'records': len(items), 'hits': sum(row['hit'] for row in items)}

    evidence_paths = ('data/frozen/tasks.json', 'training/generated-candidate/manifest.json',
        'training/generated-candidate/examples.jsonl', 'evaluation/heldout/tasks.json',
        'evaluation/heldout/manifest.json', 'experiments/heldout/h14-fp16-result.json',
        'experiments/runtime/h15-lora-qualification.json',
        'experiments/runtime/h15-independent-checkpoint-verification.json',
        'hala/contracts/responses.py', 'hala/inference/vision_candidate.py',
        'evaluation/heldout_protocol.py', 'training/scripts/convert_sft.py')
    return {
        'schema': 'hala.h18.adaptation-audit.v1',
        'reviewed_historical_snapshot_verified': True,
        'evidence_sha256': {path: sha256_file(root / path) for path in evidence_paths},
        'source': {'records': len(source['tasks']), 'splits': counts(source['tasks'], 'split'),
            'languages_by_split': {split: counts([row for row in source['tasks']
                if row['split'] == split], 'language') for split in ('train', 'dev', 'test')},
            'families_by_split': {split: counts([row for row in source['tasks']
                if row['split'] == split], 'family_id') for split in ('train', 'dev', 'test')},
            'rights_notice': source['rights_notice'], 'original_identity_validation': True},
        'candidate': {'records': len(rows), 'splits': counts(rows, 'split'),
            'languages': counts(rows, 'language'), 'families': counts(rows, 'family'),
            'completion_types': dict(sorted(Counter(strict_json(row['completion'])['type']
                for row in rows).items())),
            'completion_target_refs': dict(sorted(Counter(
                strict_json(row['completion']).get('element_ref',
                    strict_json(row['completion']).get('element_ref_hint'))
                for row in rows).items())),
            'click_protocol': 'window pixel points with image_scale and element_ref_hint; not normalized 0..1000',
            'mind_plans': 0, 'normalized_points': 0,
            'goal_contains_fixture_id': sum(row['id'] in eligible[row['id']]['goal'] for row in rows),
            'observed_controls_in_prompt': sum('Observed controls: ' in row['prompt'] for row in rows),
            'label_scope': manifest['label_scope'],
            'arabic_image_training': manifest['arabic_image_training'],
            'rights_cleared': manifest['rights_cleared'],
            'media_rights': manifest['media'][0]['provenance'],
            'media_sha256': {item['path']: item['sha256'] for item in manifest['media']},
            'every_media_hash_and_rights_validated': True,
            'first_action_source_binding_validated': True},
        'h14': {**summary(rescored), 'training_allowed': False,
            'languages': {lang: summary([row for row in rescored if row['language'] == lang])
                for lang in ('en', 'ar')},
            'families': {family: summary([row for row in rescored if row['family'] == family])
                for family in sorted({row['family'] for row in rescored})},
            'scope': 'Frozen one-click synthetic selection only; dependent authored variants.'},
        'h15': {'independent_comparison_exact': independent['exact'],
            'adapter_quality_qualified': h15['adapter_quality_qualified'],
            'arabic_remediation_claimed': h15['arabic_remediation_claimed'],
            'retained_quality_adapter': False,
            'scope': 'Two English microbatch engineering steps; remote runtime expired; no retained quality adapter.'},
        'hard_stops': ['mind_plan_training_missing', 'normalized_type_training_missing',
            'bilingual_semantic_training_missing', 'qualified_arabic_rendering_missing',
            'compatible_independent_dev_test_missing', 'prospective_data_loader_and_runner_unimplemented',
            'prospective_experiment_review_and_runtime_qualification_required'],
        'training_ready': False, 'training_executed': False,
        'quality_qualified': False,
        'config_validated': config is not None,
    }
