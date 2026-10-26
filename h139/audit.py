"""Owner independent H139 screenshot corpus quality and leakage report."""
import json
from collections import Counter
from pathlib import Path
from h139.registry import UPSTREAM,SCHEMA,check_split_manifest,sha

SPLITS={"train":64,"dev":16,"blind_test":8}

def qualify(root,old_manifests=()):
    root=Path(root)
    all_hash=set();outputs={}
    forbidden=set()
    for path in old_manifests:
        doc=json.loads(Path(path).read_text("utf8"))
        forbidden.update(x["image_sha256"] for x in doc["cases"])
    for split,amount in SPLITS.items():
        folder=root/{"train":"train64","dev":"dev16","blind_test":"blind8"}[split]
        doc=json.loads((folder/"manifest.json").read_text("utf8"))
        if doc.get("schema")!=SCHEMA or doc.get("split")!=split:
            raise ValueError("H139 expected exact frozen split schema")
        rows=check_split_manifest(doc,forbidden_sha=forbidden)
        if len(rows)!=amount:raise ValueError("H139 exact source count missing")
        per=Counter((r["app"],r["lang"]) for r in rows)
        for row in rows:
            if (row.get("image_sha256") in forbidden or
                row.get("image_sha256") in all_hash or
                row.get("real_browser_reference_state_persisted") is not True):
                raise ValueError("H139 duplicate historic image or fake source gold gesture")
            if sha((folder/row["image"]).read_bytes())!=row["image_sha256"]:
                raise ValueError("H139 original Chromium PNG SHA drift")
            all_hash.add(row["image_sha256"])
        if split=="blind_test":
            if per!={("ood-board","en"):4,("ood-board","ar"):4}:
                raise ValueError("H139 React OOD language balance")
        else:
            for app in ("js-board","alt-static-board"):
                for language in ("en","ar"):
                    if per[(app,language)]!=amount//4:
                        raise ValueError("H139 source app and AR/EN balance")
            if any(row["app"]=="ood-board" for row in rows):
                raise ValueError("H139 React blind model training leakage")
        outputs[split]={"source_manifest_sha256":sha((folder/"manifest.json").read_bytes()),
                        "cases":len(rows),"per_app_language":dict((a+"__"+l,n) for (a,l),n in per.items())}
    return outputs
