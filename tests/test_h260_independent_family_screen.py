"""H260 invented synthetic-only source-screen tests; no original app bytes."""
import unittest
from h260.independent_family_screen import scan_invented_text, require, FAMILIES


class InventedSourceScreenTests(unittest.TestCase):
    def test_plain_invented_markup(self):
        d = scan_invented_text("<html><button>Invented</button></html>",
                               "body { color: blue; }", "const x = 1;")
        self.assertEqual(d["external_url_references"], 0)
        self.assertEqual(d["css_url_references"], 0)
        self.assertEqual(d["media_tags"], 0)

    def test_remote_reference_flagged(self):
        d = scan_invented_text('<script src="https://example.invalid/a.js"></script>',
                               "", "")
        self.assertGreater(d["external_url_references"], 0)

    def test_css_media_reference_flagged(self):
        d = scan_invented_text("<html></html>",
                               "body { background: url(fake.png); }", "")
        self.assertGreater(d["css_url_references"], 0)

    def test_css_import_is_flagged(self):
        d = scan_invented_text("<html></html>",
                               '@import "invented.css";', "")
        self.assertGreater(d["external_url_references"], 0)

    def test_inline_svg_is_not_proven_rights(self):
        d = scan_invented_text("<svg><path d='M0 0'/></svg>", "", "")
        self.assertEqual(d["inline_svg_tags"], 1)

    def test_media_element_is_flagged(self):
        d = scan_invented_text('<img src="invented.png">', "", "")
        self.assertEqual(d["media_tags"], 1)

    def test_families_are_separately_identified(self):
        self.assertEqual(len(FAMILIES), 2)
        self.assertNotEqual(FAMILIES["dacheson"][0], FAMILIES["dinnocenzo"][0])

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            require(False)


if __name__ == "__main__":
    unittest.main()
