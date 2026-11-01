import unittest
from h345_familywise_gate import familywise_paired_gate


class FamilywiseGateTests(unittest.TestCase):
    def test_all_six_wins(self):
        a = familywise_paired_gate(tuple((f'family{i}', False, True) for i in range(6)))
        self.assertEqual(a['two_sided_exact_sign_p'], 0.03125)
        self.assertEqual((a['wins'], a['losses'], a['ties']), (6, 0, 0))
        self.assertFalse(a['release_allowed'])
        self.assertFalse(a['model_gain_claim_allowed'])

    def test_symmetric_all_six_losses(self):
        a = familywise_paired_gate(tuple((f'family{i}', True, False) for i in range(6)))
        self.assertEqual(a['two_sided_exact_sign_p'], 0.03125)
        self.assertEqual((a['wins'], a['losses']), (0, 6))
        self.assertFalse(a['model_gain_claim_allowed'])

    def test_balanced_discordance_is_not_evidence(self):
        rows = tuple((f'family{i}', i >= 4, i < 4) for i in range(8))
        a = familywise_paired_gate(rows)
        self.assertEqual((a['wins'], a['losses']), (4, 4))
        self.assertEqual(a['two_sided_exact_sign_p'], 1.0)

    def test_ties_do_not_inflate_sign_test(self):
        a = familywise_paired_gate(tuple((f'family{i}', True, True) for i in range(100)))
        self.assertEqual(a['two_sided_exact_sign_p'], 1.0)
        self.assertEqual(a['ties'], 100)
        self.assertFalse(a['blind_ood_verified'])

    def test_ties_cannot_rescue_insufficient_discordance(self):
        rows = tuple((f'tie{i}', False, False) for i in range(50))
        rows += (('win', False, True), ('loss', True, False))
        a = familywise_paired_gate(rows)
        self.assertEqual(a['two_sided_exact_sign_p'], 1.0)
        self.assertEqual((a['wins'], a['losses'], a['ties']), (1, 1, 50))

    def test_refuse_invalid(self):
        cases = (
            (), [('a', False, True)] * 6,
            (('a', 1, True),) * 6,
            tuple(('same', False, True) for _ in range(6)),
            tuple((f'f{i}', False, True, 'leak') for i in range(6)),
            tuple((f'f{i}', False, True) for i in range(5)),
            tuple((f'f{i}', False, True) for i in range(5)) + (('with space', False, True),),
            tuple((f'f{i}', False, True) for i in range(5)) + (('é', False, True),),
            tuple((f'f{i}', False, True) for i in range(5)) + (('x' * 49, False, True),),
        )
        for data in cases:
            with self.subTest(data=str(data)[:80]):
                with self.assertRaises(ValueError):
                    familywise_paired_gate(data)


if __name__ == '__main__':
    unittest.main()
