import unittest
from training.scripts.baseline import parse_answer, run


class NormalizedProtocolTests(unittest.TestCase):
    def test_explicit_protocol_converts_both_axes(self):
        self.assertEqual(parse_answer('{"point":[500,500]}','normalized_1000'),
                         {'point':[320,180]})
        self.assertEqual(parse_answer('```json\n{"point":[1000,1000]}\n```','normalized_1000'),
                         {'point':[639,359]})

    def test_pixel_protocol_stays_strict(self):
        with self.assertRaises(ValueError):
            parse_answer('```json\n{"point":[100,100]}\n```')
        with self.assertRaises(ValueError):
            parse_answer('{"point":[660,588]}')

    def test_ambiguous_or_malformed_output_rejects(self):
        for raw in ('before {"point":[1,2]}',
                    '```json\n{"point":[1,2]}\n``` trailing',
                    '{"point":[1,2],"point":[3,4]}',
                    '{"point":[true,2]}','{"point":[1001,0]}',
                    '{"point":[-1,0]}','{"point":[NaN,0]}',
                    '{"point":[1,2],"execute":true}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_answer(raw,'normalized_1000')

    def test_unknown_protocol_rejects_before_runtime_import(self):
        with self.assertRaises(ValueError):
            run(coordinate_mode='automatic')
