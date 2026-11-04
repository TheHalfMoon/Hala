"""Owner source-verified counterfactual 192-pair integration test.

Requires H421_OWNER_LABEL_ROOT pointing at owner PRIVATE Kaggle dataset
materialization; generic hosted CI intentionally skips original labels.
Using gold predictions here tests the SCORER contract only, not model skill.
"""
import json
import os
from pathlib import Path
import unittest

from h420.counterfactual_score import load_sources,calculate


class H421RealSourcePairedMetricIntegration(unittest.TestCase):
    def test_exact_owner_source_pairs_and_scoring_arithmetic(self):
        folder=os.environ.get("H421_OWNER_LABEL_ROOT")
        if not folder:
            self.skipTest("Original privately staged 384-case eval data not mounted in CI")
        root=Path(folder)
        pairs=load_sources(
           root/"h421_original_same_image_two_instructions.jsonl",
           root/"h421_source_counterfactual_receipt.json",
           root/"receipt.sha256")
        self.assertEqual(len(pairs),192)
        # Deterministic oracle predictions must prove the scorer is
        # internally consistent; this is NOT a genuine model evaluation.
        rows=[case for i in range(192) for case in
              (pairs[i][0],pairs[i][1])]
        pred=[{"id":row["id"],"image_sha256":row["image_sha256"],
               "raw_prediction":json.dumps(row["native_drag_target"],
                                            separators=(",",":"))}
              for row in rows]
        scores=calculate(rows,pred)
        self.assertEqual(scores["both_actions_correct"],192)
        self.assertEqual(scores["single_action_correct"],384)
        self.assertEqual(scores["collapsed_same_raw_action"],0)
        self.assertEqual(scores["wrong_instruction_source_target"],0)
        self.assertFalse(scores["independent_external_ood"])
        # Predict first target for both prompts of each same JPEG image:
        # must fail all 192 counterfactual pairs, not fake 50% success.
        collapsed=[]
        for i in range(192):
            first=pairs[i][0]["native_drag_target"]
            for j in (0,1):
                row=pairs[i][j]
                collapsed.append({"id":row["id"],
                    "image_sha256":row["image_sha256"],
                    "raw_prediction":json.dumps(first,separators=(",",":"))})
        fail=calculate(rows,collapsed)
        self.assertEqual(fail["both_actions_correct"],0)
        self.assertEqual(fail["single_action_correct"],192)
        self.assertEqual(fail["collapsed_same_raw_action"],192)

if __name__=="__main__":
    unittest.main()
