"""H127 offline viewer and aggregate source-screening CLI (no model calls)."""
import argparse
import json
from pathlib import Path
from h127.review_gate import read_json, grade
from h127.viewer import build_viewer


def main():
    parser = argparse.ArgumentParser(description="Private H127 blind GUI review.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("viewer", help="Generate offline blind reviewer page.")
    prepare.add_argument("--pack", required=True)
    prepare.add_argument("--output", required=True)
    score = commands.add_parser("score", help="Score two independently acquired submissions.")
    for flag in ("truth", "review-a", "review-b", "output"):
        score.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    if args.command == "viewer":
        result = build_viewer(args.pack, args.output)
    else:
        result = grade(read_json(args.truth), read_json(args.review_a),
                       read_json(args.review_b))
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf8") as out:
            json.dump(result, out, sort_keys=True, indent=2)
            out.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
