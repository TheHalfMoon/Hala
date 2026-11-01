import unittest
from h345_familywise_gate import familywise_paired_gate

class FamilywiseGateTests(unittest.TestCase):
    def test_all_six_wins(self):
        a = familywise_paired_gate(tuple((f'family{i}', False, True) for i in range(6)))
        self.assertEqual(a['two_sided_exact_sign_p'], 0.03125)
        self.assertFalse(a['release_allowed'])
    def test_ties_do_not_inflate_sign_test(self):
        a = familywise_paired_gate(tuple((f'family{i}', True, True) for i in range(100)))
        self.assertEqual(a['two_sided_exact_sign_p'], 1.0)
        self.assertEqual(a['ties'], 100)
    def test_refuse_invalid(self):
        for data in [(), (( 'a', 1, True),)*6, tuple(('same', False, True) for _ in range(6)), tuple((f'f{i}', False, True, 'leak') for i in range(6))]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    familywise_paired_gate(data)
if __name__ == '__main__':
    unittest.main()
