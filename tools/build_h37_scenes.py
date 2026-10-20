"""Create-only synthetic webpages for capture in an actual local Chromium browser."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from h37.scenes import cases, html_for_case, scenes_sha256, model_projection


def create(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError("H37 creates an entirely new directory only")
    (output / "pages").mkdir(parents=True, exist_ok=False)
    scenes = cases()
    html_hashes = {}
    for scene in scenes:
        text = html_for_case(scene)
        payload = text.encode("utf-8")
        path = output / "pages" / (scene["id"] + ".html")
        with path.open("xb") as stream:
            stream.write(payload)
        html_hashes[scene["id"]] = hashlib.sha256(payload).hexdigest()
    (output / "scenes.json").write_bytes(
        (json.dumps(scenes, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode("utf-8"))
    (output / "model_projection.json").write_bytes(
        (json.dumps(model_projection(scenes), ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode("utf-8"))
    meta = {"schema": "hala.h37.browser_authored_scenes.v1",
            "scenes_sha256": scenes_sha256(), "html_sha256": html_hashes,
            "source": "first-party HTML/CSS only; not an external application",
            "model_requests": 0, "split": {"dev": 32, "test": 32},
            "external_resources": False, "native_actions_executed": False}
    (output / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2)+"\n",
                                       encoding="utf-8", newline="\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obj = create(args.output)
    print(json.dumps({"cases": 64, "dev": 32, "test": 32,
                      "sha256": obj["scenes_sha256"], "output": str(args.output)}))
