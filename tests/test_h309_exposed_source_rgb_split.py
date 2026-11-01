"""H309 invented RGB negatives and opt-in real public source browsers only."""
from io import BytesIO
import os
from pathlib import Path
import unittest

from PIL import Image, PngImagePlugin

from h309.exposed_source_rgb_split import (
    ExposedSplitRegistry, SplitAuditRefused,
    audit_original_exposed_rgb_split, model_visible_rgb_fingerprint,
)


def invented_png(rgb=(20, 30, 40), alpha=255, metadata=False, size=(960, 720)):
    stream = BytesIO()
    info = None
    if metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text("invented", "metadata-only")
    Image.new("RGBA", size, (*rgb, alpha)).save(
        stream, format="PNG", pnginfo=info)
    return stream.getvalue()


class H309InventedOnlyTests(unittest.TestCase):
    def test_alpha_only_difference_is_not_model_visible(self):
        self.assertEqual(
            model_visible_rgb_fingerprint(invented_png(alpha=255)),
            model_visible_rgb_fingerprint(invented_png(alpha=7)))

    def test_png_metadata_difference_is_not_model_visible(self):
        self.assertEqual(
            model_visible_rgb_fingerprint(invented_png()),
            model_visible_rgb_fingerprint(invented_png(metadata=True)))

    def test_wrong_viewport_and_truncated_fail_closed(self):
        for data in (invented_png(size=(800, 600)),
                     invented_png()[:120], b"not a png"):
            with self.subTest(size=len(data)):
                with self.assertRaises(SplitAuditRefused):
                    model_visible_rgb_fingerprint(data)

    def test_cross_role_identical_rgb_refused_and_sealed(self):
        reg = ExposedSplitRegistry()
        for i in range(24):
            reg.add_train_candidate(invented_png(rgb=(i, 12, 24)))
        with self.assertRaises(SplitAuditRefused):
            reg.add_exposed_dev_research(
                invented_png(rgb=(0, 12, 24), alpha=4, metadata=True))
        with self.assertRaises(SplitAuditRefused):
            reg.add_exposed_dev_research(invented_png(rgb=(100, 12, 24)))

    def test_distinct_invented_roles_are_not_admitted(self):
        reg = ExposedSplitRegistry()
        for i in range(24):
            reg.add_train_candidate(invented_png(rgb=(i, 12, 24)))
        for i in range(12):
            reg.add_exposed_dev_research(invented_png(rgb=(i+60, 12, 24)))
        proof = reg.finish()
        self.assertEqual(proof["distinct_model_visible_rgb_screens"], 36)
        self.assertFalse(proof["official_train_admitted"])
        self.assertFalse(proof["independent_blind_ood"])
        self.assertFalse(proof["can_release"])
        self.assertNotIn("hashes", proof)
        with self.assertRaises(SplitAuditRefused):
            reg.add_train_candidate(invented_png())

    def test_incomplete_roles_refused(self):
        reg = ExposedSplitRegistry()
        reg.add_train_candidate(invented_png())
        with self.assertRaises(SplitAuditRefused):
            reg.finish()


@unittest.skipUnless(os.environ.get("HALA_H309_BROWSER_QUALIFY") == "1",
                     "opt-in genuine public source browser only")
class H309RealSourceBrowserTests(unittest.TestCase):
    def test_original_24_train_and_12_dev_rgb_split(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            channel = os.environ.get("HALA_H309_BROWSER_CHANNEL")
            browser = playwright.chromium.launch(
                headless=True, **({"channel": channel} if channel else {}))
            try:
                result = audit_original_exposed_rgb_split(
                    browser, Path("h285/shiftdesk.html"),
                    Path("h296/fieldport.html"))
            finally:
                browser.close()
        self.assertEqual(result["public_train_candidates"], 24)
        self.assertEqual(result["public_exposed_dev_research_screens"], 12)
        self.assertEqual(result["distinct_model_visible_rgb_screens"], 36)
        self.assertFalse(result["official_dev_admitted"])
        self.assertFalse(result["independent_blind_ood"])
        self.assertEqual(result["model_generations"], 0)


if __name__ == "__main__":
    unittest.main()
