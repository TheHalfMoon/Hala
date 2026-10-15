"""H18 audits evidence; a prospective specification never authorizes training."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from training.adaptation import audit, validate_config
from training.artifacts import sha256_file
from evaluation.heldout_protocol import score

ROOT = Path(__file__).resolve().parents[1]


class AdaptationTests(unittest.TestCase):
    def test_real_inventory_does_not_mistake_actions_for_plans_or_normalized_points(self):
        report = audit(ROOT)
        self.assertEqual(report['candidate']['records'], 58)
        self.assertEqual(report['candidate']['languages'], {'en': 58})
        self.assertEqual(report['candidate']['completion_types'],
                         {'click': 20, 'semantic_invoke': 20, 'set_value': 18})
        self.assertEqual(report['candidate']['mind_plans'], 0)
        self.assertEqual(report['candidate']['normalized_points'], 0)
        self.assertEqual(report['candidate']['goal_contains_fixture_id'], 58)
        self.assertEqual(report['source']['splits'], {'dev': 32, 'test': 32, 'train': 116})
        self.assertEqual(report['h14']['hits'], 30)
        self.assertEqual(report['h14']['languages']['ar']['hits'], 14)
        self.assertEqual(report['h14']['families']['transit_arrivals']['hits'], 1)
        self.assertEqual(report['h14']['families']['warehouse_stock']['hits'], 2)
        self.assertEqual(report['h14']['families']['document_versions']['hits'], 2)
        self.assertFalse(report['training_ready'])
        self.assertEqual(audit(ROOT), report)

    def test_existing_synthetic_dev_test_are_not_claimed_as_compatible_sft(self):
        report = audit(ROOT)
        self.assertEqual(report['candidate']['splits'], {'train': 58})
        self.assertIn('compatible_independent_dev_test_missing', report['hard_stops'])
        self.assertIn('bilingual_semantic_training_missing', report['hard_stops'])
        self.assertIn('mind_plan_training_missing', report['hard_stops'])
        self.assertIn('normalized_type_training_missing', report['hard_stops'])

    def fixture(self, temp):
        target = Path(temp)
        for relative in ('training/generated-candidate', 'data/frozen',
                         'evaluation/heldout', 'experiments/heldout', 'experiments/runtime'):
            shutil.copytree(ROOT / relative, target / relative)
        for relative in ('hala/contracts/responses.py', 'hala/inference/vision_candidate.py',
                         'evaluation/heldout_protocol.py', 'training/scripts/convert_sft.py'):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        return target

    def mutate_row(self, root, change, update_digest):
        path = root / 'training/generated-candidate/examples.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        change(rows[0])
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        if update_digest:
            manifest_path = path.with_name('manifest.json')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['dataset_sha256'] = sha256_file(path)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    def test_edited_bytes_rejected_even_with_recomputed_manifest_digest(self):
        for update in (False, True):
            with self.subTest(updated_digest=update), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                self.mutate_row(root, lambda row: row.update(prompt='Invented compatible task'), update)
                with self.assertRaises(ValueError):
                    audit(root)

    def test_relabeling_h14_or_arabic_does_not_create_admitted_training(self):
        changes = ({'id': 'h14-transit_arrivals-00'}, {'language': 'ar'}, {'split': 'dev'},
                   {'completion': '{"point":[500,500]}'})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                self.mutate_row(root, lambda row: row.update(change), True)
                with self.assertRaises(ValueError):
                    audit(root)

    def test_rights_and_media_evidence_are_checked_not_just_counted(self):
        for kind in ('rights', 'media'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                base = root / 'training/generated-candidate'
                path = base / 'manifest.json'
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if kind == 'rights':
                    manifest['media'][0]['provenance']['commercial_training_allowed'] = False
                    path.write_text(json.dumps(manifest), encoding='utf-8')
                else:
                    (base / manifest['media'][0]['path']).write_bytes(b'altered')
                with self.assertRaises(ValueError):
                    audit(root)

    def test_h14_scores_are_recomputed_and_train_exclusion_cannot_be_relabeled(self):
        for kind in ('score', 'exclusion'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                relative = ('experiments/heldout/h14-fp16-result.json' if kind == 'score'
                            else 'evaluation/heldout/tasks.json')
                path = root / relative
                value = json.loads(path.read_text(encoding='utf-8'))
                if kind == 'score':
                    value['cases'][0]['hit'] = not value['cases'][0]['hit']
                else:
                    value['training_allowed'] = True
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit(root)

    def test_same_family_screenshot_swap_cannot_change_original_image_label_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            base = root / 'training/generated-candidate'
            path = base / 'examples.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            first, second = rows[0], rows[1]
            self.assertEqual(first['family'], second['family'])
            for key in ('image_path', 'image_sha256'):
                first[key], second[key] = second[key], first[key]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
            manifest['dataset_sha256'] = sha256_file(path)
            (base / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(root)

    def test_coherent_image_bytes_and_catalog_hash_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            base = root / 'training/generated-candidate'
            manifest_path = base / 'manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            item = manifest['media'][0]
            from PIL import Image
            Image.new('RGB', (800, 600), 'red').save(base / item['path'], format='PNG')
            item['sha256'] = sha256_file(base / item['path'])
            # Update the actual corresponding row, not just the first row by assumption.
            path = base / 'examples.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            for row in rows:
                if row['image_path'] == item['path']:
                    row['image_sha256'] = item['sha256']
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            manifest['dataset_sha256'] = sha256_file(path)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(root)

    def test_fabricated_h14_perfect_raw_outputs_and_consistent_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            cases = {case['id']: case for case in json.loads(
                (root / 'evaluation/heldout/tasks.json').read_text(encoding='utf-8'))['cases']}
            path = root / 'experiments/heldout/h14-fp16-result.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            for row in report['cases']:
                case = cases[row['id']]
                control = next(item for item in case['controls'] if item['id'] == case['expected_target'])
                left, top, right, bottom = control['box']
                width, height = case['size']
                raw = json.dumps({'point': [round((left + right) * 500 / width),
                                            round((top + bottom) * 500 / height)]})
                row.update(output=raw, **score(case, raw))
                self.assertTrue(row['hit'])
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(root)

    def test_false_h15_exact_or_coherently_rehashed_quality_report_is_rejected(self):
        for kind in ('exact_string', 'paired_report'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                path = root / 'experiments/runtime/h15-independent-checkpoint-verification.json'
                independent = json.loads(path.read_text(encoding='utf-8'))
                if kind == 'exact_string':
                    independent['exact'] = 'FULL QUALITY CERTIFIED'
                else:
                    report_path = root / 'experiments/runtime/h15-lora-qualification.json'
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    report['adapter_quality_qualified'] = True
                    report_path.write_text(json.dumps(report), encoding='utf-8')
                    independent['report_sha256'] = sha256_file(report_path)
                path.write_text(json.dumps(independent), encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit(root)

    def test_false_manifest_scope_exclusion_and_raqm_claims_are_rejected(self):
        for key, value in (('label_scope', 'Fully validated bound Mind plans'),
                           ('excluded_records', 0), ('raqm_qualified', True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                root = self.fixture(temp)
                path = root / 'training/generated-candidate/manifest.json'
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if key == 'label_scope':
                    manifest[key] = value
                else:
                    manifest['arabic_image_training'][key] = value
                path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit(root)

    def test_coherently_rewritten_h14_task_and_freeze_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            path = root / 'evaluation/heldout/tasks.json'
            value = json.loads(path.read_text(encoding='utf-8'))
            value['cases'][0]['goal'] = 'Substituted evaluation goal'
            path.write_text(json.dumps(value), encoding='utf-8')
            manifest_path = path.with_name('manifest.json')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['files']['evaluation/heldout/tasks.json'] = sha256_file(path)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(root)

    def test_prospective_shared_config_cannot_enable_jobs_or_separate_adapters(self):
        config = json.loads((ROOT / 'experiments/adaptation/shared-multitask.json').read_text())
        validate_config(config)
        for key, value in (('training_enabled', True), ('download_weights', True),
                           ('adapter_count', 2), ('base_revision', 'main'),
                           ('datasets', ['evaluation/heldout/tasks.json']),
                           ('training_ready', True), ('adapter_count', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = copy.deepcopy(config)
                changed[key] = value
                validate_config(changed)
        report = audit(ROOT, config)
        self.assertFalse(report['training_ready'])
        self.assertFalse(report['training_executed'])


if __name__ == '__main__':
    unittest.main()
