"""H26 CPU-only prospective grammar gates, no model inference or task success."""
import copy
import json
import unittest

from h26 import choice
from h25 import baseline


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, code, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("BOS/special tokens not permitted")
        return [ord(c) for c in code]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(c) for c in ids)


class BrokenTokenizer(FakeTokenizer):
    def encode(self, code, add_special_tokens=False):
        return [67, 49]


class H26ChoiceTests(unittest.TestCase):
    def test_exact_eight_codes_seven_canonical_typed_pairs(self):
        self.assertEqual(set(choice.CHOICES),{f"C{i}" for i in range(8)})
        self.assertIsNone(choice.advisory("C0"))
        self.assertEqual(len({tuple(x) for x in choice.CHOICES.values() if x}),7)
        for code in list(choice.CHOICES)[1:]:
            typed = choice.advisory(code)
            self.assertEqual(typed["kind"], "next")
            self.assertEqual(baseline.typed_proposal(json.dumps(typed)),typed)

    def test_refuse_untrusted_raw_repairs_from_h24_h25(self):
        invalid = (
            "", " C1", "C1\n", "c1", "C8", "C01", "visual",
            '{"kind":"visual","tool":"visual","subgoal":"locate_verified_target"}',
            '{"kind":"next","tool":"visual","subgoal":"locate_verified_target"}',
            "C1 because it looks right", None, 1, {}, ["C1"],
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                choice.advisory(value)

    def test_every_expected_h25_oracle_has_exact_one_code(self):
        pairs=baseline.expected_rows()
        self.assertEqual(len(pairs),42)
        self.assertEqual({choice.reference_code(expected) for _,expected,_ in pairs},
                         set(choice.CHOICES)-{"C0"})

    def test_all_reference_codes_score_independently_from_grammar(self):
        for row, expected, state in baseline.expected_rows():
            code=choice.reference_code(expected)
            s=choice.score(code,expected)
            self.assertTrue(s["format_valid"])
            self.assertTrue(s["semantic_correct"])
            self.assertFalse(s["withheld"])
            self.assertEqual(s["suggestion"], expected)
            wrong = "C1" if code!="C1" else "C7"
            miss=choice.score(wrong,expected)
            self.assertTrue(miss["format_valid"])
            self.assertFalse(miss["semantic_correct"])
            self.assertFalse(miss["withheld"])
            self.assertFalse(choice.score("C0",expected)["semantic_correct"])
            self.assertTrue(choice.score("C0",expected)["withheld"])
            self.assertFalse(choice.score("C1 blah",expected)["format_valid"])

    def test_reject_oracle_drift_or_unknown_pair(self):
        for obj in (
            {"kind":"visual","tool":"visual","subgoal":"locate_verified_target"},
            {"kind":"next","tool":"shell","subgoal":"run_as_admin"},
            {"kind":"next","tool":"visual","subgoal":"reject_external_command"},
            {"kind":"next","tool":"visual","subgoal":"locate_verified_target","approved":True},
        ):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                choice.reference_code(obj)

    def test_payload_excludes_tracking_and_references(self):
        for row,expected,state in baseline.expected_rows():
            payload=json.loads(choice.user_payload(row))
            self.assertEqual(set(payload),{"goal","locale","observation"})
            self.assertNotIn("oracle_private",str(payload))
            self.assertNotIn("expected",str(payload))
            self.assertNotIn("source_ref",str(payload))
            self.assertNotIn(row["source_ref"]["id"],str(payload))
        with self.assertRaises(ValueError):
            choice.user_payload({"model_input":{},"source_ref":{},"oracle_private":{}})

    def test_finite_state_token_gate_blocks_alternative_words(self):
        gate=choice.TokenGate.build(FakeTokenizer(),[20,21,22])
        self.assertEqual(gate.allowed(0,[20,21,22]),[67])
        self.assertEqual(gate.allowed(0,[20,21,22,67]),[48,49,50,51,52,53,54,55])
        for n in range(8):
            self.assertEqual(gate.allowed(0,[20,21,22,67,48+n]),[99])
        self.assertEqual(gate.max_new_tokens,3)

    def test_finite_state_gate_fail_closed_on_hijacked_prefix(self):
        gate=choice.TokenGate.build(FakeTokenizer(),[20,21,22])
        with self.assertRaises(ValueError):
            gate.allowed(0,[20,99,22])
        with self.assertRaises(ValueError):
            gate.allowed(1,[20,21,22])
        self.assertEqual(gate.allowed(0,[20,21,22,120]),[99])

    def test_truncated_or_corrupted_code_never_accepted(self):
        gate=choice.TokenGate.build(FakeTokenizer(),[20,21,22])
        for bad in ("C","C7\n","C999","C0 ","  C2"):
            with self.assertRaises(ValueError):
                choice.raw_choice(bad)
        self.assertEqual(gate.allowed(0,[20,21,22,67,55,55]),[99])

    def test_tokenizer_lossless_and_prefix_free(self):
        gate=choice.TokenGate.build(FakeTokenizer(),[20,21,22])
        self.assertEqual(len(gate.words),8)
        with self.assertRaises(ValueError):
            choice.TokenGate.build(BrokenTokenizer(),[20,21,22])
        with self.assertRaises(ValueError):
            choice.TokenGate.build(FakeTokenizer(),[])

    def test_no_os_or_native_authority_side_effects(self):
        self.assertEqual(choice.advisory("C7"),
                         {"kind":"next","tool":"abstain","subgoal":"reject_external_command"})
        self.assertIsNone(choice.advisory("C0"))
        self.assertTrue(all("execute" not in item and "authorized" not in item
                            for item in (choice.advisory("C1"),choice.advisory("C7"))))


if __name__=="__main__":
    unittest.main()
