"""python -m evaluation --provider oracle --split test --output report.json"""
import argparse
import json
from pathlib import Path

from data.factory import load_frozen
from .engine import OracleProvider, ControlProvider, MockProvider, run_case
from .metrics import summarize


def main():
    parser = argparse.ArgumentParser(description="Offline synthetic infrastructure evaluation; no real model inference")
    parser.add_argument("--provider", choices=["oracle", "mock", "wrong_target", "stale", "invalid", "false_done"], default="oracle")
    parser.add_argument("--suite", choices=["all", "intent", "gui"], default="all")
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screenshots", type=Path, help="Optional PNG output directory; requires Pillow and a Unicode font")
    parser.add_argument("--font")
    args = parser.parse_args()
    manifest = load_frozen()
    cases = [c for c in manifest["tasks"] if (args.suite == "all" or c["suite"] == args.suite) and (args.split == "all" or c["split"] == args.split)]
    results, images = [], []
    for case in cases:
        provider = OracleProvider(case) if args.provider == "oracle" else MockProvider() if args.provider == "mock" else ControlProvider(args.provider, case)
        results.append(run_case(case, provider))
        if args.screenshots:
            from .render import render_case
            images.append(render_case(case, args.screenshots / (case["task_id"] + ".png"), args.font))
    report = {"manifest_revision": manifest["revision"], "seed": manifest["seed"], "provider": args.provider, "selected_split": args.split, "metrics": summarize(results), "cases": results, "screenshots": images}
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
