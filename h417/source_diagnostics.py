"""Inspect source-image ID diversity versus actual content SHA diversity.

Pure offline manifest analysis; never downloads or prints raw screenshots.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path

def diagnose(path):
    path=Path(path)
    rows=[json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
    groups=defaultdict(list)
    for r in rows:
        groups[r["image_sha256"]].append({
            "id":r.get("id"),"image_path":r.get("image_path"),"domain":r.get("family")
        })
    dups=[v for v in groups.values() if len(v)>1]
    return {"rows":len(rows),
            "distinct_source_image_sha256":len(groups),
            "duplicate_image_sha256_groups":len(dups),
            "duplicated_rows":sum(len(v)-1 for v in dups),
            "sample_id_collisions":dups[:8],
            "unique_source_ids":len({r["id"] for r in rows}),
            "unique_paths":len({r["image_path"] for r in rows})}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",required=True)
    args=p.parse_args()
    print(json.dumps(diagnose(args.jsonl),indent=2,sort_keys=True))
