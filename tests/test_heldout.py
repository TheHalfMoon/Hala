import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import heldout_fixtures as fixtures
from evaluation import heldout_runner as runner
from evaluation.heldout_protocol import parse, score, model_prompt, summarize, wilson


def center_output(case, control):
    # Find an exactly representable normalized pixel inside each visible control.
    l, t, r, b = control['box']
    w, h = case['size']
    return json.dumps({'point': [((l + r) // 2 * 1000 + w - 1) // w,
                                 ((t + b) // 2 * 1000 + h - 1) // h]})


class HeldoutProtocolTests(unittest.TestCase):
    def test_generalized_mapping_edges_and_half_open_rectangles(self):
        self.assertEqual(parse('{"point":[500,500]}', 1024, 768), (512, 384))
        self.assertEqual(parse('{"point":[1000,1000]}', 800, 600), (799, 599))
        self.assertEqual(parse('{"point":[0,0]}', 1, 1), (0, 0))
        case = {'size': [1000, 1000], 'expected_target': 'target',
                'controls': [{'id': 'target', 'box': [100, 100, 200, 200]}]}
        self.assertTrue(score(case, '{"point":[100,100]}')['hit'])
        self.assertFalse(score(case, '{"point":[200,150]}')['hit'])
        self.assertFalse(score(case, '{"point":[150,200]}')['hit'])

    def test_strict_schema_rejects_untrusted_and_false_completion(self):
        bad = ['```\n{"point":[1,2]}\n```', '```JSON\n{"point":[1,2]}\n```',
               '```json {"point":[1,2]}\n```', '```json\r\n{"point":[1,2]}\r\n```',
               'before ```json\n{"point":[1,2]}\n```',
               '```json\n{"point":[1,2]}\n``` trailing',
               '```json\n{"point":[1,2]}\n```\n```json\n{"point":[3,4]}\n```',
               '```json\n{"point":[1,2]}\n```\n{"point":[3,4]}',
               '{"point":[1,2],"done":true}',
               '{"point":[1,2],"point":[2,3]}', '{"point":[true,2]}',
               '{"point":[1.0,2]}', '{"point":[NaN,2]}', '{"point":[Infinity,2]}',
               '{"point":[null,2]}', '{"point":[1001,2]}', '{"point":[-1,2]}',
               '{"point":[1,2,3]}', 'I completed the task.', '[1,2]', 'null',
               '{"point":{"x":1,"y":2}}', '{"point":[1,2]} trailing', 'x' * 4097]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw, 800, 600)
        for dimensions in [(True, 600), (0, 600), (800, 4097), (800.0, 600)]:
            with self.assertRaises(ValueError):
                parse('{"point":[1,2]}', *dimensions)

    def test_calibrated_exact_fence_preserves_strict_coordinate_semantics(self):
        self.assertEqual(parse('```json\n{"point":[500,1000]}\n```', 1024, 768), (512, 767))
        self.assertEqual(parse(' \n```json\n{"point":[0,500]}\n```\n ', 800, 600), (0, 300))
        case = fixtures.author_cases()['cases'][0]
        target = next(c for c in case['controls'] if c['id'] == case['expected_target'])
        raw = center_output(case, target)
        self.assertEqual(score(case, raw), score(case, '```json\n' + raw + '\n```'))
        for raw in ('{"point":[true,2]}', '{"point":[NaN,2]}',
                    '{"point":[1,2],"point":[3,4]}', '{"point":[1,2],"done":true}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse('```json\n' + raw + '\n```', 800, 600)

    def test_dpi_metadata_never_changes_image_frame(self):
        case = fixtures.author_cases()['cases'][0]
        control = next(c for c in case['controls'] if c['id'] == case['expected_target'])
        raw = center_output(case, control)
        expected = score(case, raw)
        for dpi in (96, 120, 144, 192, 384):
            self.assertEqual(score({**case, 'dpi': dpi}, raw), expected)

    def test_negative_providers_wrong_invalid_and_false_done_all_fail(self):
        for case in fixtures.author_cases()['cases']:
            target = next(c for c in case['controls'] if c['id'] == case['expected_target'])
            wrong = next(c for c in case['controls'] if c['id'] != case['expected_target'])
            oracle = score(case, center_output(case, target))
            self.assertTrue(oracle['hit'] and oracle['fixture_complete'])
            adversary = score(case, center_output(case, wrong))
            self.assertTrue(adversary['wrong_target'])
            self.assertFalse(adversary['fixture_complete'])
            for raw in ('{"point":[false,0]}', '{"done":true}',
                        center_output(case, target)[:-1] + ',"done":true}'):
                result = score(case, raw)
                self.assertFalse(result['schema_valid'] or result['fixture_complete'])
            miss = score(case, '{"point":[0,0]}')
            self.assertTrue(miss['miss'])
            self.assertFalse(miss['fixture_complete'])

    def test_private_annotations_cannot_change_model_prompt(self):
        case = fixtures.author_cases()['cases'][0]
        changed = copy.deepcopy(case)
        changed['expected_target'] = 'PRIVATE_TERMINAL_SECRET'
        changed['controls'] = [{'box': [1, 2, 3, 4], 'id': 'PRIVATE_BOX_SECRET'}]
        changed['terminal_state'] = 'PRIVATE_STATE_SECRET'
        self.assertEqual(model_prompt(case), model_prompt(changed))
        self.assertNotIn('PRIVATE_', model_prompt(changed))

    def test_descriptive_metrics_include_invalid_in_denominator(self):
        rows = []
        for case in fixtures.author_cases()['cases']:
            rows.append({'family': case['family'], 'language': case['language'],
                         **score(case, '{"done":true}')})
        metrics = summarize(rows)
        self.assertEqual(metrics['overall']['count'], 64)
        self.assertEqual(metrics['languages']['ar']['count'], 32)
        self.assertEqual(len(metrics['families']), 8)
        self.assertEqual(metrics['overall']['fixture_complete']['rate'], 0)
        self.assertEqual(wilson(0, 0), None)
        self.assertAlmostEqual(wilson(0, 64)[1], 0.056624, places=5)


class HeldoutFreezeTests(unittest.TestCase):
    def test_frozen_original_balanced_specs_match_authoring(self):
        dataset = fixtures.validate(json.loads(fixtures.FROZEN.read_text(encoding='utf-8')))
        self.assertEqual(dataset, fixtures.author_cases())
        self.assertEqual(dataset['rights']['license'], 'CC0-1.0')
        self.assertFalse(dataset['training_allowed'])
        prior = json.loads((fixtures.ROOT / 'data/frozen/tasks.json').read_text(encoding='utf-8'))
        prior_text = json.dumps(prior, ensure_ascii=False)
        for case in dataset['cases']:
            self.assertNotIn(case['id'], prior_text)
            self.assertNotIn(case['goal'], prior_text)
        self.assertEqual(len({tuple(c['size']) for c in dataset['cases']}), 4)

    def test_trusted_freeze_verifies_and_detects_tamper(self):
        expected = fixtures.digest(fixtures.MANIFEST)
        fixtures.verify_freeze(expected)
        with self.assertRaises(ValueError):
            fixtures.verify_freeze('0' * 64)
        original_digest = fixtures.digest
        with patch.object(fixtures, 'digest', side_effect=lambda p:
                          'f' * 64 if Path(p) == fixtures.FROZEN else original_digest(p)):
            with self.assertRaises(ValueError):
                fixtures.verify_freeze(expected)

    def test_invalid_and_overlapping_geometry_rejected(self):
        case = fixtures.author_cases()
        case['cases'][0]['controls'][0]['box'] = [-1, 0, 1, 1]
        with self.assertRaises(ValueError):
            fixtures.validate(case)
        case = fixtures.author_cases()
        case['cases'][0]['controls'][1]['box'] = case['cases'][0]['controls'][0]['box']
        with self.assertRaises(ValueError):
            fixtures.validate(case)

    def test_remote_guard_blocks_before_import_or_download(self):
        with patch('evaluation.heldout_runner.platform.system', return_value='Windows'):
            with self.assertRaises(ValueError):
                runner.run('x', 'uncreated-output', 'font.ttf')
        with patch('evaluation.heldout_runner.platform.system', return_value='Linux'), \
                patch.dict(os.environ, {'HALA_REMOTE_HELDOUT_ALLOWED': '1',
                                        'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}):
            with self.assertRaises(ValueError):
                runner.remote_guard()

    def test_arabic_without_raqm_fails_closed(self):
        with patch('PIL.features.check', return_value=False):
            with self.assertRaisesRegex(ValueError, 'RAQM'):
                fixtures.render(fixtures.author_cases()['cases'][4], 'not-a-font')

    def test_all_english_labels_fit_actual_local_renderer(self):
        font = Path('C:/Windows/Fonts/arial.ttf')
        if not font.exists():
            font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        if not font.exists():
            self.skipTest('No local test font')
        for case in fixtures.author_cases()['cases']:
            if case['language'] == 'en':
                with self.subTest(case=case['id']):
                    image = fixtures.render(case, font)
                    self.assertEqual(image.size, tuple(case['size']))
                    image.close()

    def test_actual_renderer_border_matches_half_open_scoring(self):
        font = Path('C:/Windows/Fonts/arial.ttf')
        if not font.exists():
            font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        if not font.exists():
            self.skipTest('No local test font')
        case = fixtures.author_cases()['cases'][0]
        image = fixtures.render(case, font)
        l, t, r, b = case['controls'][0]['box']
        self.assertEqual(image.getpixel((r - 1, t + 3)), (52, 68, 85))
        self.assertEqual(image.getpixel((r, t + 3)), (237, 241, 244))
        self.assertEqual(image.getpixel((l + 3, b - 1)), (52, 68, 85))
        self.assertEqual(image.getpixel((l + 3, b)), (237, 241, 244))
        image.close()
