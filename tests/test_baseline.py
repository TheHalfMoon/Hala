import unittest

from training.scripts.baseline import check_point, parse_answer


class BaselineCheckerTests(unittest.TestCase):
    def test_raw_answers_reject_duplicate_keys_and_nonfinite_constants(self):
        self.assertEqual(parse_answer('{"point":[15,25]}'), {'point': [15, 25]})
        for raw in ('{"point":[0,0],"point":[15,25]}',
                    '{"point":[NaN,25]}', '{"point":[Infinity,25]}',
                    '{"point":[15,25],"success":true}', 'x' * 4097):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                parse_answer(raw)

    def test_target_checker_uses_half_open_independent_geometry(self):
        self.assertTrue(check_point({'point': [15, 25]}, [10, 20, 30, 40]))
        self.assertFalse(check_point({'point': [30, 25]}, [10, 20, 30, 40]))
        self.assertFalse(check_point({'point': [15, 40]}, [10, 20, 30, 40]))

    def test_malformed_and_authority_fields_are_rejected(self):
        for value in ({'point': [True, 25]}, {'point': [15.5, 25]},
                      {'point': [15, 25], 'success': True}, {'point': '15,25'},
                      [15, 25], {'point': [15]}):
            self.assertFalse(check_point(value, [10, 20, 30, 40]))


if __name__ == '__main__':
    unittest.main()
