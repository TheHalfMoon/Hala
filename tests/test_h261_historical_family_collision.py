"""Invented synthetic H261 fixtures, with unittest assertions surviving Python -O."""
import unittest

from h261.historical_family_collision import (
    audit, compare, require, shingles, similarity, tree_entries,
)


def fixture_inventory(source, media, tokens):
    return {"sources": set(source), "media": set(media),
            "shapes": {str(i): shingles(t.encode()) for i, t in enumerate(tokens)}}


class H261SyntheticTests(unittest.TestCase):
    def test_git_tree_parser_synthetic(self):
        raw = b"100644 blob " + b"a" * 40 + b"\tui/example.js\0"
        self.assertEqual(tree_entries(raw), [(".js", "a" * 40)])

    def test_non_source_is_ignored(self):
        raw = b"100644 blob " + b"b" * 40 + b"\tLICENSE\0"
        self.assertEqual(tree_entries(raw), [])

    def test_malformed_missing_tab(self):
        with self.assertRaises(ValueError):
            tree_entries(b"bad\0")

    def test_malformed_commit_not_blob(self):
        with self.assertRaises(ValueError):
            tree_entries(b"100644 commit " + b"a" * 40 + b"\tx.js\0")

    def test_malformed_blob_identifier(self):
        with self.assertRaises(ValueError):
            tree_entries(b"100644 blob BAD\tx.js\0")

    def test_source_token_overlap(self):
        text = ("function invented timer set mode short break render display; " * 8).encode()
        self.assertEqual(similarity(shingles(text), shingles(text)), 1.0)

    def test_short_generic_code_not_evidence(self):
        self.assertEqual(shingles(b"let x=1;"), frozenset())
        self.assertEqual(similarity(frozenset(), frozenset()), 0.0)

    def test_pairwise_exact_media_and_source_detection(self):
        sample = ["function timer start pause shortBreak display value; " * 8]
        left = fixture_inventory(["a", "b"], ["m1"], sample)
        right = fixture_inventory(["b", "c"], ["m1", "m2"], sample)
        result = compare(left, right)
        self.assertEqual(result["identical_source_blobs"], 1)
        self.assertEqual(result["identical_media_blobs"], 1)
        self.assertEqual(result["source_token_pairs_ge_90pct"], 1)

    def test_different_source_not_90pct(self):
        left = fixture_inventory(["a"], [], ["function alpha input submit button text value; " * 8])
        right = fixture_inventory(["b"], [], ["const random chart render plot scale legend color; " * 8])
        self.assertEqual(compare(left, right)["source_token_pairs_ge_90pct"], 0)

    def test_missing_family_denied_before_file_access(self):
        with self.assertRaises(ValueError):
            audit({"h250": (None, "invented")})

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            require(False)


if __name__ == "__main__":
    unittest.main()
