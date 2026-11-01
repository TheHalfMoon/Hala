"""H330: design packet integrity, NOT proof of an EGM Goldfish review.

These CPU checks cannot establish that a separate subagent ran, had no
parent context, reviewed the supplied source, or approved the design.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "designs" / "hala-rtl-visual-action-learning-egm.md"
GUIDE = ROOT / "docs" / "egm" / "HALA.md"
AGENTS = ROOT / "AGENTS.md"
CLAUDE = ROOT / "CLAUDE.md"
UPSTREAM_PIN = "403d8881d62022490637d740657b33f01e321ea8"


class H330DesignContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = SPEC.read_text(encoding="utf-8")
        cls.guide = GUIDE.read_text(encoding="utf-8")
        cls.agents = AGENTS.read_text(encoding="utf-8")
        cls.claude = CLAUDE.read_text(encoding="utf-8")

    def test_pin_and_upstream_review_standard(self):
        for document in (self.spec, self.guide):
            self.assertIn("google/elephant-goldfish-model", document)
            self.assertIn(UPSTREAM_PIN, document)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", UPSTREAM_PIN))
        for term in ("feature-spec-standard.md",
                     "implementation-readiness-standard.md"):
            self.assertIn(term, self.spec)
            self.assertIn(term, self.guide)

    def test_design_contains_testable_problem_and_alternatives(self):
        for term in ("## 1. Problem", "## 2. Goals",
                     "## 3. Roles", "## 4. Candidate architecture",
                     "## 5. Alternatives", "## 6. Preregistered",
                     "## 7. Failure", "## 8. Implementation phases",
                     "## 9. Design questions"):
            self.assertIn(term, self.spec)
        for alternative in ("Tighten Qwen prompt", "LoRA on independently cleared",
                            "Hybrid: deterministic UI geometry"):
            self.assertIn(alternative, self.spec)

    def test_safety_exclusions_and_ungated_training_remain_explicit(self):
        for document in (self.spec, self.guide):
            for term in ("H199", "H173", "H293/H294",
                         "TRAIN", "Goldfish"):
                self.assertIn(term, document)
            self.assertRegex(document, r"(not|NOT).*review")
        self.assertIn("REVIEW_BLOCKED", self.guide)
        self.assertIn("REVIEW_BLOCKED", self.agents)
        self.assertIn("NO_GO", self.spec)
        self.assertIn("INCONCLUSIVE", self.spec)

    def test_fresh_context_and_no_false_review_attestation(self):
        self.assertIn("fresh isolated", self.guide)
        self.assertIn("NO parent conversation", self.guide)
        self.assertIn("NOT REVIEWED", self.spec)
        self.assertIn("must not** be marked as a Goldfish review", self.spec)
        self.assertIn("Goldfish reviewers", self.agents)
        self.assertIn("REVIEW_BLOCKED", self.claude)
        self.assertIn("docs/egm/HALA.md", self.claude)
        self.assertIn("No upstream files are copied", self.guide)

    def test_only_local_repo_artifact_links_and_no_embedding_model(self):
        self.assertIn("docs/designs/hala-rtl-visual-action-learning-egm.md",
                      self.guide)
        self.assertIn("docs/egm/HALA.md", self.agents)
        self.assertIn("No EGM component should be imported", self.guide)
        self.assertIn("does not itself make Hala more accurate", self.guide)


if __name__ == "__main__":
    unittest.main()
