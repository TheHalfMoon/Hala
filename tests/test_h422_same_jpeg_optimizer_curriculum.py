"""H422 same-JPEG two-native-prompt optimizer-step curriculum.

These intentionally fake cases test batch composition only; they are NOT
training evidence. Original 32k source JPEG bytes and next real GPU stage
have independent private source and receipt gates.
"""
from collections import Counter
import unittest

from h422.kaggle import h422_two_prompt_native_lora as m


def synthetic_cases():
    rows=[]
    for j in range(m.H422_ORIGINAL_SCREEN_N):
        img=f"{j:064x}"
        domain=("slider","slide_resize","text_highlight")[j%3]
        for k in range(2):
            rows.append({
                "id":f"synthetic-{j}-{k}",
                "image_sha256":img,
                "domain":domain,
                "kind":"original","language":"en",
                "body":f"Drag original handle to {100+500*k}",
                "start":[100,200],"end":[200+500*k,200],
                "ordered":True,
            })
    for j in range(m.H422_AR_TRAIN_N):
        rows.append({
            "id":f"synthetic-browser-ar-{j}",
            "image_sha256":f"{16000+j:064x}",
            "domain":"slider","kind":"firstparty","language":"ar",
            "body":"اسحب المقبض","start":[100,200],
            "end":[600,200],"ordered":True,
        })
    return rows


class H422SameJPEGOptimizerSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=synthetic_cases()

    def test_8976_optimizer_steps_pair_real_instruction_contrasts(self):
        order=m.paired_step_curriculum(self.rows)
        self.assertEqual(len(order),35904)
        self.assertEqual(len(order)//4,8976)
        self.assertEqual(order,m.paired_step_curriculum(self.rows))
        seen=Counter(order)
        self.assertTrue(all(seen[i]==1 for i in range(32000)))
        self.assertTrue(all(seen[i]==4 for i in range(32000,len(self.rows))))
        paired_blocks=arabic_blocks=0
        for j in range(0,len(order),4):
            block=[self.rows[i] for i in order[j:j+4]]
            language={c["language"] for c in block}
            images=Counter(c["image_sha256"] for c in block)
            if language=={"en"}:
                paired_blocks+=1
                self.assertEqual(sorted(images.values()),[2,2])
                self.assertEqual(len(images),2)
                for image in images:
                    alternatives=[x for x in block if x["image_sha256"]==image]
                    self.assertNotEqual(alternatives[0]["body"],alternatives[1]["body"])
                    self.assertNotEqual(alternatives[0]["end"],alternatives[1]["end"])
            else:
                arabic_blocks+=1
                self.assertEqual(language,{"ar"})
                self.assertEqual(len(images),4)
        self.assertEqual(paired_blocks,8000)
        self.assertEqual(arabic_blocks,976)

    def test_missing_one_original_native_instruction_rejected(self):
        rows=self.rows[:-1]
        with self.assertRaisesRegex(ValueError,"32976"):
            m.paired_step_curriculum(rows)
        rows=list(self.rows)
        rows[1]={**rows[1],"image_sha256":"f"*64}
        with self.assertRaisesRegex(ValueError,"16K pair images|two REAL different tasks"):
            m.paired_step_curriculum(rows)

    def test_identical_end_target_or_same_body_rejected(self):
        rows=list(self.rows)
        rows[1]={**rows[1],"end":rows[0]["end"]}
        with self.assertRaisesRegex(ValueError,"two REAL different tasks"):
            m.paired_step_curriculum(rows)
        rows=list(self.rows)
        rows[1]={**rows[1],"body":rows[0]["body"]}
        with self.assertRaisesRegex(ValueError,"two REAL different tasks"):
            m.paired_step_curriculum(rows)

    def test_arabic_repeats_require_distinct_original_browser_screenshots(self):
        rows=list(self.rows)
        rows[32001]={**rows[32001],
                     "image_sha256":rows[32000]["image_sha256"]}
        with self.assertRaisesRegex(ValueError,"Arabic TRAIN source repeated"):
            m.paired_step_curriculum(rows)
        rows=list(self.rows)
        rows[32000]={**rows[32000],"kind":"original"}
        with self.assertRaisesRegex(ValueError,"original browser source"):
            m.paired_step_curriculum(rows)

    def test_untrained_source_only_batch_plan_has_no_model_result_claim(self):
        self.assertEqual(m.H422_OPT_STEPS,8976)
        self.assertEqual(m.H422_GRAD_EXAMPLES,35904)
        self.assertEqual(m.GRAD_ACCUM,4)
        self.assertEqual(m.H422_AR_REPEAT+1,4)
        self.assertEqual(m.H422_ORIGINAL_SCREEN_N//2,8000)
        self.assertEqual(m.H422_DONE,
            "COMPLETE_PRIVATE_H422_TRUE_TWO_PROMPT_NATIVE_DRAG")

if __name__=="__main__":
    unittest.main()
