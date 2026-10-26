"""Strict independent original MIT dnd-kit GUI source pilot gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h140.dndkit_pilot import SOURCE_SHA,native_geometry_proposal,qualified_build,own_board


class H140DndKitOriginalTests(unittest.TestCase):
    def test_original_board_has_unique_cards_and_true_source(self):
        board,card_id,title=own_board(12,9,5)
        self.assertIn(card_id,board["columns"][0]["cardIds"])
        self.assertNotIn(card_id,board["columns"][1]["cardIds"])
        self.assertEqual(board["cards"][card_id]["title"],title)
        self.assertEqual(len(board["cards"]),19)

    def test_original_position_schedule_is_multigeometry(self):
        rows=[native_geometry_proposal(i) for i in range(64)]
        self.assertEqual({r[4] for r in rows},{"en","ar"})
        self.assertEqual(sum(r[4]=="en" for r in rows),32)
        self.assertEqual({r[0] for r in rows},set(range(6,12)))
        self.assertGreaterEqual(len({(r[1],r[2]) for r in rows}),12)
        for n,s,d,scroll,lang in rows:
            self.assertTrue(6<=n<=11 and 0<=s<n and 0<=d<n and 0<=scroll<190)

    def test_source_sha_license_tracked_clean_and_original_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"LICENSE").write_text("MIT License\nCopyright source author",encoding="utf8")
            (root/"dist").mkdir()
            (root/"dist"/"index.html").write_text("<!DOCTYPE html>",encoding="utf8")
            with patch("h140.dndkit_pilot.subprocess.check_output",side_effect=[SOURCE_SHA+"\n",""]):
                self.assertEqual(qualified_build(root),root/"dist")
            with patch("h140.dndkit_pilot.subprocess.check_output",side_effect=[SOURCE_SHA+"\n","changed tracked app"]):
                with self.assertRaises(ValueError):
                    qualified_build(root)


if __name__=="__main__":
    unittest.main()
