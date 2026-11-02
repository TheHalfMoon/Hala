"""H371: bounded original synthetic browser repeat qualifier.

Captures raw unittest logs in memory but NEVER prints them, private screenshots,
tasks, scene IDs, point coordinates, gold, exception messages or stack traces.
No model weights, model inference, TRAIN admission or blind OOD.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from h370.browser_failure_stage import STAGES
from h373.screenshot_error_buckets import CAPTURE_BUCKETS

CASES = (
    ("h319", "test_h319_original_dynamic_bilingual_browser.py", 9),
    ("h337", "test_h337_source_miss_distance_bands.py", 7),
)
SUMMARY = re.compile(r"Ran (\d+) tests? in [\d.]+s")
CODES = re.compile(
    r"(H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED|"
    r"H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED) "
    r"stage=([a-z_]+) kind=([a-z_]+)(?: capture=([a-z_]+))?"
)
ALLOWED_KINDS = {
    "value", "assertion", "type", "key", "index", "runtime",
    "timeout", "os", "other", "browser_timeout", "browser_error",
}


def scrub_failure(output: str, flow: str) -> str:
    """Return only allowlisted stage and exception class from raw logs."""
    for prefix, stage, kind, capture in CODES.findall(output):
        if (stage in STAGES[flow] and kind in ALLOWED_KINDS
                and prefix.startswith(flow.upper())):
            safe = "stage=" + stage + " kind=" + kind
            if (capture in CAPTURE_BUCKETS and
                    kind in {"browser_error", "browser_timeout"} and
                    stage.endswith(("_image_capture", "_post_policy_frame"))):
                safe += " capture=" + capture
            return safe
    return "stage=unclassified kind=unclassified"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", required=True, choices=("msedge", "chromium"))
    parser.add_argument("--repeats", required=True, type=int)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 4:
        raise ValueError("H371_BAD_REPEAT_BOUND")
    env = os.environ.copy()
    env.update({
        "HALA_H319_BROWSER_QUALIFY": "1",
        "HALA_H319_BROWSER_CHANNEL": args.browser,
        "HALA_H337_SOURCE_BROWSER": "1",
        "HALA_H337_BROWSER_CHANNEL": args.browser,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    total = 0
    for trial in range(1, args.repeats + 1):
        for flow, source_file, expected in CASES:
            cmd = [
                sys.executable, "-B", "-m", "unittest", "discover",
                "-s", "tests", "-p", source_file, "-q",
            ]
            try:
                done = subprocess.run(
                    cmd, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=240,
                    check=False)
            except subprocess.TimeoutExpired:
                print("H371_RESEARCH_ONLY FAIL flow=" + flow
                      + " trial=" + str(trial)
                      + " stage=timeout kind=timeout", flush=True)
                raise SystemExit(1)
            # Never emit raw captured stdout/stderr to CI. Extract ONLY
            # a count and fixed H370 redacted stage/kind on failure.
            output = done.stdout + done.stderr
            matches = SUMMARY.findall(output)
            valid = done.returncode == 0 and matches == [str(expected)]
            if not valid:
                print("H371_RESEARCH_ONLY FAIL flow=" + flow
                      + " trial=" + str(trial) + " "
                      + scrub_failure(output, flow), flush=True)
                raise SystemExit(1)
            total += expected
            print("H371_RESEARCH_ONLY PASS flow=" + flow
                  + " trial=" + str(trial)
                  + " tests=" + str(expected), flush=True)
    print("H371_RESEARCH_ONLY ALL_PASS repetitions=" + str(args.repeats)
          + " source_only_tests=" + str(total)
          + " rights_admitted=false blind_ood=false learned_gain=false",
          flush=True)


if __name__ == "__main__":
    main()
