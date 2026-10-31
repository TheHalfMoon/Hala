"""H262 public source-only checks. No screenshots, labels, or private receipts."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "h262" / "stockroom.html"


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = {}
        self.csp = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        self.attributes.setdefault(tag, []).append(attrs)
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy":
            self.csp.append(attrs.get("content", ""))


class OriginalStockroomSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.parser = AuditParser()
        cls.parser.feed(cls.source)

    def test_source_is_standalone_utf8_html(self):
        self.assertTrue(self.source.startswith("<!doctype html>"))
        self.assertIn('<meta charset="utf-8">', self.source)
        self.assertEqual(self.parser.tags.count("html"), 1)

    def test_no_external_media_or_embedded_third_party_assets(self):
        forbidden = {"img", "svg", "link", "iframe", "video", "audio", "canvas", "object", "embed"}
        self.assertTrue(forbidden.isdisjoint(self.parser.tags))
        for attrs in self.parser.attributes.values():
            for entry in attrs:
                self.assertFalse(any(key in entry for key in ("src", "href", "srcset")))
        self.assertNotRegex(self.source, r"https?://|//cdn\.|@import")

    def test_strict_offline_csp(self):
        self.assertEqual(len(self.parser.csp), 1)
        for policy in ("default-src 'none'", "connect-src 'none'", "img-src 'none'",
                       "font-src 'none'", "base-uri 'none'", "form-action 'none'"):
            self.assertIn(policy, self.parser.csp[0])

    def test_only_inline_first_party_script_and_style(self):
        self.assertEqual(self.parser.tags.count("script"), 1)
        self.assertEqual(self.parser.tags.count("style"), 1)
        self.assertNotRegex(self.source, r"\b(?:fetch|XMLHttpRequest|WebSocket|sendBeacon)\s*\(")

    def test_bilingual_rtl_support(self):
        self.assertIn('en:{title:"Stockroom"', self.source)
        self.assertIn('ar:{title:"المخزن"', self.source)
        self.assertIn('document.documentElement.dir=', self.source)
        self.assertIn('toLocaleLowerCase()', self.source)

    def test_native_click_and_drag_state_changes(self):
        self.assertIn('addEventListener("click"', self.source)
        self.assertIn('addEventListener("dragstart"', self.source)
        self.assertIn('addEventListener("dragover"', self.source)
        self.assertIn('addEventListener("drop"', self.source)
        self.assertIn('item.count+=amount', self.source)

    def test_native_text_and_keyboard_actions(self):
        self.assertIn('addEventListener("submit"', self.source)
        self.assertIn('addEventListener("input"', self.source)
        self.assertIn('addEventListener("keydown"', self.source)
        self.assertIn('e.key==="Escape"', self.source)

    def test_scrollable_history_and_accessible_status(self):
        self.assertIn('overflow-y:auto', self.source)
        self.assertIn('id="activity" tabindex="0"', self.source)
        self.assertIn('role="status"', self.source)

    def test_no_persistent_storage_or_analytics(self):
        for term in ("localStorage", "sessionStorage", "indexedDB", "navigator.sendBeacon",
                     "document.cookie", "google-analytics", "gtag("):
            self.assertNotIn(term, self.source)

    def test_user_supplied_text_escaped_before_dom_render(self):
        self.assertIn("const safe=s=>", self.source)
        self.assertIn("safe(nameOf(item))", self.source)
        self.assertIn("safe(message)", self.source)

    def test_no_model_or_task_success_claims_in_source(self):
        for term in ("accuracy", "model improvement", "OOD success", "trained weights"):
            self.assertNotIn(term, self.source.lower())


if __name__ == "__main__":
    unittest.main()
