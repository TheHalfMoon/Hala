import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('h20_diagnostic', Path(__file__).resolve().parents[1] / 'tools/diagnose_reliance.py')


class RelianceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(cls.m)

    def test_encoding_controls(self):
        p = self.m.payload('احفظ الملف')
        import json
        self.assertEqual(json.loads(self.m.encode(p, True)), p)
        self.assertEqual(json.loads(self.m.encode(p, False)), p)
        self.assertNotEqual(self.m.encode(p, True), self.m.encode(p, False))
        corrupted = 'الحفظ'.encode('utf-8').decode('cp1252', errors='surrogateescape')
        self.assertEqual(json.loads(self.m.encode(self.m.payload(corrupted), True))['state'], corrupted)

    def test_split_families_and_language_balance(self):
        train, test = self.m.corpus()
        self.assertFalse({x['family'] for x in train} & {x['family'] for x in test})
        for rows in (train, test):
            self.assertEqual(sum(x['language'] == 'en' for x in rows), sum(x['language'] == 'ar' for x in rows))
        with self.assertRaises(ValueError):
            self.m.check_splits(train, train)

    def test_invalid_response_and_authority(self):
        for obj in ({}, {'answers': {'route': {'choice': 'shell'}}}, {'answers': {'route': {'choice': 'mind', 'probabilities': {'mind': float('nan')}}}}):
            with self.assertRaises(ValueError):
                self.m.route_answer(obj)
        self.assertEqual(self.m.deterministic({'structured': True, 'visual': False, 'ambiguous': False, 'unsafe': False}), 'deterministic')
        self.assertEqual(self.m.deterministic({'structured': True, 'visual': True, 'ambiguous': True, 'unsafe': False}), 'abstain')

    def test_small_classifier_and_unknown_abstention(self):
        train, _ = self.m.corpus()
        model = self.m.fit(train)
        self.assertEqual(self.m.predict(model, ''), 'abstain')
        self.assertIn(self.m.predict(model, train[0]['state']), self.m.ROUTES)
        self.assertEqual(self.m.predict(model, 'unseenxyz'), 'abstain')

    def test_bounded_json_huge_integer_probability_is_rejected(self):
        import json
        text = '{"answers":{"route":{"choice":"mind","probabilities":{"deterministic":0,"mind":' + '1' + '0' * 4000 + ',"type":0,"abstain":0}}}}'
        self.assertLess(len(text.encode('utf-8')), 5000)
        with self.assertRaises(ValueError):
            self.m.route_answer(json.loads(text))
        for probability in (0, 1, 0.5):
            value = {'answers': {'route': {'choice': 'mind', 'probabilities': {
                'deterministic': 1 - probability, 'mind': probability, 'type': 0, 'abstain': 0}}}}
            self.assertEqual(self.m.route_answer(value), 'mind')

    def test_recorded_transport_controls_preserve_identity(self):
        import hashlib
        import json
        path = Path(__file__).resolve().parents[1] / 'experiments/reliance/h20-direct-comparison.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        cases = record['stdio_probe']['cases']
        self.assertEqual([bool(json.loads(c['response_text'])['result'].get('isError')) for c in cases], [False, True, False, False])
        repaired = []
        for case in cases:
            wire = bytes.fromhex(case['wire_hex'])
            self.assertEqual(hashlib.sha256(wire).hexdigest(), case['wire_sha256'])
            if case['mode'] == 'explicit_utf8':
                repaired.append(json.loads(json.loads(case['response_text'])['result']['content'][0]['text']))
        self.assertEqual(repaired[0]['result']['answers'], repaired[1]['result']['answers'])
        self.assertTrue(all(r['predictions']['d1_direct_escaped_json']['raw']['status'] == 200 for r in record['comparisons']))
        self.assertEqual([r['status'] for r in record['controls']], [200, 200, 200, 200, 400, 400])
        self.assertIn('surrogate', record['controls'][-1]['response_text'])


if __name__ == '__main__':
    unittest.main()
