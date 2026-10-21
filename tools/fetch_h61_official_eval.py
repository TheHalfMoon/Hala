"""H61 fetch the upstream official public evaluation scenes for every domain.

The fetch is bounded, records every file and its digest, and never touches the
hidden test split. It is the same code path the remote run uses, so the remote
run inherits a validated fetch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h61.official_eval import apply_frame, fetch_domain, load_scenes  # noqa: E402

DOMAINS = ("slider", "sheet", "slide_resize", "text_highlight")


def main() -> int:
    root = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    summary = {}
    for domain in DOMAINS:
        target = root / domain
        if not (target / "scenes.json").exists():
            fetch_domain(domain, limit, target)
        scenes = load_scenes(target / "scenes.json")
        framed = apply_frame(scenes, target)
        ordered = sum(1 for scene in framed if scene["task"]["ordered"])
        summary[domain] = {
            "scenes": len(framed),
            "ordered": ordered,
            "unordered": len(framed) - ordered,
            "frame_sizes": sorted({(scene["image_width"], scene["image_height"])
                                   for scene in framed}),
            "sample_intent": framed[0]["task"]["intent"] if framed else None,
        }
        print(json.dumps({domain: summary[domain]}, ensure_ascii=False), flush=True)
    (root / "official-eval-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
