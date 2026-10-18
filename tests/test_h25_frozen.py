"""Immutable file-level validation of first-party H25 bilingual candidate."""
import hashlib
import json
from pathlib import Path
import unittest

from h25 import corpus as h

ROOT=Path(__file__).resolve().parents[1]/"h25"/"frozen"
PINS={
    "cases.json": "b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b",
    "manifest.json": "b4c27bba3209708d9892ddb6f4987027183a116d55421c07deca3e2d43470be3",
    "public-dev.jsonl": "5379f3db559c304c037e971cdf6aa15020211f57c42e0bb30fb75a643f947e57",
}


class H25ArchiveTests(unittest.TestCase):
    def test_exact_sha256_pins(self):
        for name, digest in PINS.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)

    def test_prospective_corpus_matches_authored_source(self):
        source=h.corpus()
        self.assertEqual((ROOT/"cases.json").read_bytes(),h.canonical(source))
        self.assertEqual((ROOT/"manifest.json").read_bytes(),h.canonical(h.validate(source)))
        self.assertEqual(h.digest(source),PINS["cases.json"])

    def test_public_dev_bytes_and_labels_are_separate(self):
        expected=b"".join(h.canonical(x) for x in h.public_dev_inputs(h.corpus()))
        self.assertEqual((ROOT/"public-dev.jsonl").read_bytes(),expected)
        parsed=[json.loads(row) for row in expected.decode("utf-8").splitlines()]
        self.assertEqual(len(parsed),42)
        self.assertEqual({r["source_ref"]["language"] for r in parsed},{"ar","en"})
        self.assertTrue(all("oracle_private" not in r["model_input"] for r in parsed))

    def test_public_dev_has_no_authored_trajectories_or_h24_test(self):
        rows=[json.loads(row) for row in (ROOT/"public-dev.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertFalse(any(r["source_ref"]["family"] in h.SPLITS["authoring"] for r in rows))
        self.assertFalse(any("h24-" in json.dumps(r) or "h22-" in json.dumps(r) for r in rows))

    def test_no_train_or_test_labeling_claim(self):
        data=json.loads((ROOT/"cases.json").read_text(encoding="utf8"))
        self.assertEqual(set(h.SPLITS),{"authoring","dev"})
        self.assertFalse(any(row["partition"] in ("test","benchmark") for row in data["cases"]))
        self.assertEqual(data["rights"]["commercial_training_clearance"],"review_required")


if __name__=="__main__":
    unittest.main()
