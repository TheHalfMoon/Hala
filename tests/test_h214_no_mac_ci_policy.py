"""H214: fail-closed source-only regression for approved CI hosts.

No owner files, screenshots, model weights, GPU or private metadata.
"""
from __future__ import annotations

from pathlib import Path
import re
import unittest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/h213-cross-platform-frozen-source-cpu.yml"
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("H214 Windows/Linux-only CI policy: " + reason)


def validate_workflow_text(text: str) -> None:
    """Enforce a fixed Windows/Linux matrix and exact source-only checks."""
    require(type(text) is str and 0 < len(text) <= 20000, "invalid workflow")
    require(not re.search(r"(?i)\b(?:macos|darwin|osx)\b", text),
            "unapproved host")
    matches = re.findall(
        r"(?m)^\s*os:\s*\[([^\]\r\n]+)\]\s*$", text
    )
    require(len(matches) == 1, "one explicit host matrix required")
    hosts = [
        value.strip().strip(chr(34)).strip(chr(39))
        for value in matches[0].split(",")
    ]
    require(hosts == ["ubuntu-24.04", "windows-2025"],
            "only pinned Windows and Linux hosts permitted")
    runners = re.findall(r"(?m)^\s*runs-on:\s*(.+?)\s*$", text)
    require(runners == ["${{ matrix.os }}"], "unapproved extra runner")
    require('python-version: ["3.11", "3.12"]' in text,
            "pinned Python matrix required")
    require("Pillow==12.3.0" in text, "pinned Pillow required")
    require("python -m h213.frozen_candidate_reproduction_gate" in text,
            "frozen source reproduction required")
    require("python -O -m h213.frozen_candidate_reproduction_gate" in text,
            "optimized frozen reproduction required")
    require(
        "python -m unittest discover -s tests -p 'test_h214_no_mac_ci_policy.py'" in text
        and "python -O -m unittest discover -s tests -p 'test_h214_no_mac_ci_policy.py'" in text,
        "normal and optimized host policy tests required",
    )


class NoMacCiPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = WORKFLOW.read_text(encoding="utf-8")

    def test_real_workflow_has_only_authorized_hosts(self) -> None:
        validate_workflow_text(self.original)

    def deny(self, text: str) -> None:
        with self.assertRaises(ValueError):
            validate_workflow_text(text)

    def test_mac_runner_rejected(self) -> None:
        self.deny(self.original.replace("windows-2025]", "windows-2025, macos-15]"))

    def test_darwin_job_rejected(self) -> None:
        self.deny(self.original + "\n  other:\n    runs-on: darwin\n")

    def test_windows_omission_rejected(self) -> None:
        self.deny(self.original.replace("ubuntu-24.04, windows-2025", "ubuntu-24.04"))

    def test_linux_omission_rejected(self) -> None:
        self.deny(self.original.replace("ubuntu-24.04, windows-2025", "windows-2025"))

    def test_unpinned_runner_rejected(self) -> None:
        self.deny(self.original.replace("runs-on: ${{ matrix.os }}", "runs-on: ubuntu-latest"))

    def test_duplicate_host_matrix_rejected(self) -> None:
        self.deny(self.original + "\n    os: [ubuntu-24.04, windows-2025]\n")

    def test_unpinned_pillow_rejected(self) -> None:
        self.deny(self.original.replace("Pillow==12.3.0", "Pillow"))

    def test_missing_optimized_policy_rejected(self) -> None:
        self.deny(self.original.replace(
            "python -O -m unittest discover -s tests -p 'test_h214_no_mac_ci_policy.py'",
            "echo skipped",
        ))

    def test_wrong_type_or_empty_rejected(self) -> None:
        self.deny("")
        self.deny(None)


if __name__ == "__main__":
    unittest.main()
