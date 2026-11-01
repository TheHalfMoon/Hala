"""H321: protect the original HTML Git blob from platform text conversion."""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import unittest

from h319.beaconboard_dynamic_research import BLOB, checked_source

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "h319" / "beaconboard.html"


class ImmutableSourceCheckoutTests(unittest.TestCase):
    def test_explicit_binary_git_attribute_for_pinned_html(self):
        rules = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        self.assertIn("h319/beaconboard.html -text", [
            line.strip() for line in rules if not line.lstrip().startswith("#")
        ])

    def test_real_original_file_has_exact_git_blob_identity(self):
        data = HTML.read_bytes()
        actual = sha1(b"blob " + str(len(data)).encode("ascii") +
                      b"\x00" + data).hexdigest()
        self.assertEqual(actual, BLOB)
        checked_source(HTML)


if __name__ == "__main__":
    unittest.main()
