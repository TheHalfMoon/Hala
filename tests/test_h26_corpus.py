"""H26 genuinely new domain DEV data, not protected unseen TEST."""
import copy
from collections import Counter
import json
import unittest

from h25 import corpus as earlier
from h26 import corpus,choice


class H26CorpusTests(unittest.TestCase):
    def setUp(self):
        self.data=corpus.dataset()

    def test_exact_reproducible_source(self):
        m=corpus.validate(self.data)
        self.assertEqual(m["cases"],42)
        self.assertEqual(m["sha256"],corpus.digest(corpus.dataset()))
        self.assertFalse(m["heldout_test"])
        self.assertFalse(m["model_run"])

    def test_domains_do_not_overlap_h25(self):
        self.assertEqual(set(corpus.AUTHORED) & set(earlier.GOALS),set())
        self.assertEqual(len(corpus.AUTHORED),3)

    def test_seven_states_per_language_per_family(self):
        states=Counter((r["source_ref"]["family"],r["source_ref"]["language"],
                        r["source_ref"]["state"]) for r in self.data["records"])
        self.assertEqual(len(states),42)
        self.assertTrue(all(x==1 for x in states.values()))
        self.assertEqual(set(corpus.STATE_TO_CODE.values()),set(choice.CHOICES)-{"C0"})

    def test_no_h25_oracle_labels_in_model_input(self):
        for row in self.data["records"]:
            model=row["model_input"]
            self.assertEqual(set(model),{"goal","locale","observation"})
            display=json.dumps(model,ensure_ascii=False)
            for word in ('"source_ref"','"private_reference"','"choice"',
                         '"state"','"expected"','"partition"'):
                self.assertNotIn(word,display)
            self.assertNotIn(row["source_ref"]["id"],display)
            self.assertNotIn(row["source_ref"]["family"],display)

    def test_unicode_arabic_and_distinct_goals(self):
        rows=self.data["records"]
        self.assertEqual(len({x["model_input"]["goal"] for x in rows}),42)
        for row in rows:
            if row["source_ref"]["language"]=="ar":
                self.assertGreaterEqual(sum(0x600<=ord(z)<=0x6ff for z in row["model_input"]["goal"]),12)
                self.assertNotIn("?",row["model_input"]["goal"])
                self.assertNotIn("\ufffd",row["model_input"]["goal"])

    def test_separated_true_hostile_page_from_user_destructive_goal(self):
        for row in self.data["records"]:
            state=row["source_ref"]["state"]
            field=row["model_input"]["observation"]["untrusted_page_text"]
            self.assertEqual(bool(field),state=="hostile_page")
            expected=row["private_reference"]["choice"]
            self.assertEqual(expected,corpus.STATE_TO_CODE[state])
            if state=="unsafe_user":
                self.assertEqual(expected,"C7")
            elif state=="hostile_page":
                self.assertEqual(expected,"C3")

    def test_frame_staleness_visible_without_rank_or_literal_state(self):
        for row in self.data["records"]:
            obs=row["model_input"]["observation"]
            stale=obs["observed_frame"]!=obs["current_frame"]
            self.assertEqual(stale,row["source_ref"]["state"]=="stale")
            self.assertEqual(len(obs["observed_frame"]),20)
            self.assertNotIn(row["source_ref"]["family"],obs["observed_frame"])
            self.assertNotIn("stale",obs["observed_frame"])

    def test_altering_dataset_or_private_oracle_fails(self):
        mutations=(
            lambda d:d["records"][0]["private_reference"].update({"choice":"C7"}),
            lambda d:d["records"][1]["model_input"].update({"expected":"C1"}),
            lambda d:d["records"][2]["source_ref"].update({"language":"fr"}),
            lambda d:d.update({"partition":"test"}),
            lambda d:d["records"].pop(),
        )
        for mutate in mutations:
            changed=copy.deepcopy(self.data)
            mutate(changed)
            with self.assertRaises(ValueError):
                corpus.validate(changed)

    def test_no_model_actions_authorized(self):
        for row in self.data["records"]:
            typed=row["private_reference"]["typed"]
            self.assertNotIn("execution_authorized",typed)
            self.assertNotIn("completed",typed)
            self.assertEqual(typed["kind"],"next")
            self.assertNotIn("device",typed)

    def test_actual_public_model_input_json_is_canonical(self):
        for raw in corpus.public_inputs(self.data):
            self.assertEqual(set(raw),{"goal","locale","observation"})
            rendered=json.loads(corpus.canon(raw))
            self.assertEqual(raw,rendered)


if __name__=="__main__":
    unittest.main()
