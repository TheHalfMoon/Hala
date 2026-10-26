"""H127 source-only safe blind review tests; all screenshots are synthetic."""
import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from h127.review_gate import (
    COUNT, EXPECTED_SCHEDULE, FAMILIES, SCHEMA, grade, make_bundle,
    validate_submission,
)
from h127.viewer import build_viewer


def synthetic_cases(root):
    from PIL import Image
    cases = []
    for i in range(COUNT):
        path = root / ("case%02d.png" % i)
        image = Image.new("RGB", (80, 80), (i*3 % 256, i*5 % 256, i*7 % 256))
        image.save(path)
        cases.append({
            "id": "synthetic-%d" % i,
            "split": "train",
            "family": FAMILIES[i//16],
            "intent": "Move START to END <script>private intent</script>" if i==0 else "Move item.",
            "image_path": path.name,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "image_width": 80,
            "image_height": 80,
            "start_bbox": [5, 5, 10, 10],
            "end_bbox": [60, 60, 65, 65],
        })
    return cases


def review(truth, reviewer, correct=True):
    return {
        "schema": SCHEMA, "pack_sha256": truth["pack_sha256"],
        "reviewer_id": reviewer, "independent_blind_attestation": True,
        "annotations": [{
            "key": c["key"], "status": "located",
            "start": [6, 6] if correct else [20, 20],
            "end": [61, 61] if correct else [20, 20]
        } for c in truth["truth"]],
    }


class H127BlindGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = synthetic_cases(self.root)
        self.pack = self.root / "blind.zip"
        self.truth_file = self.root / "PRIVATE-gold.json"

    def build(self):
        receipt = make_bundle(self.cases, self.root, EXPECTED_SCHEDULE,
                              self.pack, self.truth_file)
        return receipt, json.loads(self.truth_file.read_text("utf8"))

    def test_pack_separates_labels_and_reviews(self):
        receipt, truth = self.build()
        self.assertEqual(receipt["tasks"], 64)
        self.assertFalse(receipt["human_review_completed"])
        self.assertFalse(truth["human_review_completed"])
        with zipfile.ZipFile(self.pack) as bundle:
            names = bundle.namelist()
            self.assertEqual(len(names), 65)
            self.assertNotIn("PRIVATE-gold.json", names)
            blind = bundle.read("tasks.json")
            self.assertNotIn(b"start_bbox", blind)
            self.assertNotIn(b"end_bbox", blind)
            self.assertNotIn(b"synthetic-1", blind)
            self.assertEqual(len(json.loads(blind)["tasks"]), 64)
        self.assertNotIn("intent", json.dumps(truth))
        reviewer_view = self.root / "review-session"
        view_result = build_viewer(self.pack, reviewer_view)
        self.assertFalse(view_result["labels_embedded"])
        html = (reviewer_view/"index.html").read_text("utf8")
        self.assertNotIn("<script>private intent</script>", html)
        self.assertNotIn("start_bbox", html)
        self.assertEqual(len(list((reviewer_view/"images").iterdir())),64)

    def test_scores_only_complete_two_distinct_self_attested_reviews(self):
        _, truth = self.build()
        a = review(truth, "independent-A")
        b = review(truth, "independent-B")
        result = grade(truth, a, b)
        self.assertEqual(result["aggregate"]["both_correct"], 64)
        self.assertEqual(result["status"], "PROVISIONAL_SELF_ATTESTED_SCREEN_PASS")
        self.assertFalse(result["reviewer_identities_independently_verified"])
        self.assertFalse(result["model_skill_qualified"])
        wrong = review(truth, "independent-B", correct=False)
        self.assertEqual(grade(truth, a, wrong)["aggregate"]["both_correct"], 0)
        self.assertEqual(grade(truth, a, wrong)["status"], "FAIL_OR_AMBIGUOUS_SOURCE_IDENTIFIABILITY")
        duplicate = copy.deepcopy(a)
        with self.assertRaises(ValueError):
            grade(truth, a, duplicate)

    def test_rejects_leaks_missing_and_malformed_annotation(self):
        _, truth = self.build()
        a = review(truth, "independent-A")
        for tweak in (
            lambda x: x.update(independent_blind_attestation=False),
            lambda x: x.update(pack_sha256="0"*64),
            lambda x: x["annotations"].pop(),
            lambda x: x["annotations"][0].update(start=[-1,5]),
            lambda x: x["annotations"][0].update(end=[float("nan"),5]),
            lambda x: x["annotations"][0].update(status="ambiguous"),
        ):
            bad = copy.deepcopy(a)
            tweak(bad)
            with self.assertRaises(ValueError):
                validate_submission(bad, truth)
        abandoned = copy.deepcopy(a)
        abandoned["annotations"][0] = {
            "key": abandoned["annotations"][0]["key"], "status": "ambiguous",
            "start": None, "end": None,
        }
        validate_submission(abandoned, truth)
        self.assertEqual(grade(truth, abandoned, review(truth, "independent-B"))
                         ["aggregate"]["a_abstain"], 1)

    def test_refuses_bad_source_and_create_overwrite(self):
        with self.assertRaises(ValueError):
            make_bundle(self.cases, self.root, "b"*64, self.pack, self.truth_file)
        tampered=copy.deepcopy(self.cases)
        tampered[0]["image_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            make_bundle(tampered, self.root, EXPECTED_SCHEDULE,
                        self.pack, self.truth_file)
        tampered = copy.deepcopy(self.cases)
        tampered[0]["family"] = FAMILIES[1]
        with self.assertRaises(ValueError):
            make_bundle(tampered, self.root, EXPECTED_SCHEDULE,
                        self.pack, self.truth_file)
        self.build()
        with self.assertRaises(ValueError):
            self.build()
        first = self.root/"review-session"
        build_viewer(self.pack, first)
        with self.assertRaises(ValueError):
            build_viewer(self.pack, first)


if __name__ == "__main__":
    unittest.main()
