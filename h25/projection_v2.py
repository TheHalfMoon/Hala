"""H25 source-only reviewer projection: remove direct oracle-like state booleans.

This module DOES NOT create a held-out TEST, run GPU models, or authorize a
native action. The original published H25 v1 archive remains byte-immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path

from h25 import corpus as original

SCHEMA = "hala.h25.dev_public_projection.v2"
H25_PIN = "b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b"
TEXT = {
    "seminar_facilitation": {
        "en": ("Review revised seminar abstracts", "Past session listings"),
        "ar": ("مراجعة ملخصات الندوات المعدلة", "قوائم الجلسات السابقة"),
    },
    "accessibility_review": {
        "en": ("Keyboard navigation audit", "Earlier audit reports"),
        "ar": ("تدقيق التنقل بلوحة المفاتيح", "تقارير التدقيق السابقة"),
    },
    "habitat_survey": {
        "en": ("Wetland bird observation form", "Archived field reports"),
        "ar": ("نموذج رصد طيور المنطقة الرطبة", "تقارير الميدان المؤرشفة"),
    },
}


def token(case_id, context):
    # Purpose is to remove plain-text family/rank/state from frame IDs, NOT
    # to claim secrecy or an unseen benchmark in publicly released samples.
    digest = hashlib.sha256((case_id + "|" + context).encode("utf-8")).hexdigest()
    return "f" + digest[:19]


def candidate():
    frozen = Path(__file__).resolve().parent / "frozen" / "cases.json"
    raw = frozen.read_bytes()
    if hashlib.sha256(raw).hexdigest() != H25_PIN:
        raise ValueError("Original H25 source archive SHA256 differs from frozen pin")
    data = json.loads(raw)
    if original.canonical(data) != raw or original.validate(data)["sha256"] != H25_PIN:
        raise ValueError("Original H25 source provenance or canonical bytes invalid")
    rows = []
    seen = set()
    for r in data["cases"]:
        if r["partition"] != "dev":
            continue
        if r["family"] not in TEXT or r["language"] not in ("ar", "en"):
            raise ValueError("Unreviewed H25 DEV family or language")
        private = r["oracle_private"]
        is_stale = private["state"] == "stale"
        current = token(r["id"], "current")
        observed = token(r["id"], "observed") if is_stale else current
        if current in seen or observed in seen:
            raise ValueError("Duplicated frame identity across case family")
        seen.add(current)
        seen.add(observed) if observed != current else None
        labels = TEXT[r["family"]][r["language"]]
        input_record = {
            "goal": r["goal"],
            "locale": r["locale"],
            "observation": {
                "observed_frame": observed,
                "current_frame": current,
                "controls": [
                    {"role": "button", "label": labels[0], "bounds": r["observed"]["ui_controls"][0]["bounds"]},
                    {"role": "button", "label": labels[1], "bounds": r["observed"]["ui_controls"][1]["bounds"]},
                ],
                "untrusted_page_text": r["observed"]["page_untrusted_text"],
            },
        }
        rows.append({
            "source_ref": {"id": r["id"], "family": r["family"], "language": r["language"]},
            "model_input": input_record,
        })
    if len(rows) != 42:
        raise ValueError("All prospective H25 DEV source rows are required")
    return {
        "schema": SCHEMA,
        "source_cases_sha256": H25_PIN,
        "version": "projection-v2-no-oracle-bits",
        "model_proposals_authorized": False,
        "model_baseline_measured": False,
        "heldout_test": False,
        "rows": rows,
    }


def validate(value):
    if original.canonical(value) != original.canonical(candidate()):
        raise ValueError("H25 projection source/labels/rows differ from reviewed source")
    rows = value["rows"]
    if len(rows) != 42 or len({r["source_ref"]["id"] for r in rows}) != 42:
        raise ValueError("Missing or duplicate H25 DEV case")
    for row in rows:
        x = row["model_input"]
        if set(x) != {"goal", "locale", "observation"}:
            raise ValueError("Unapproved H25 prompt fields")
        if set(x["observation"]) != {"observed_frame", "current_frame", "controls", "untrusted_page_text"}:
            raise ValueError("Unexpected H25 oracle-like input metadata")
        for ref in ("observed_frame", "current_frame"):
            text = x["observation"][ref]
            if not (len(text) == 20 and text.startswith("f") and
                    all(ch in "0123456789abcdef" for ch in text[1:])):
                raise ValueError("Predictable family/rank/state text in frame ID")
        for name in (row["source_ref"]["family"], row["source_ref"]["language"], row["source_ref"]["id"]):
            if name in x["observation"]["observed_frame"] or name in x["observation"]["current_frame"]:
                raise ValueError("Frame label leaked source reference")
    return {
        "schema": SCHEMA,
        "source_cases_sha256": H25_PIN,
        "source_splits": "DEV only; public non-holdout",
        "rows": 42,
        "languages": {"en": 21, "ar": 21},
        "source_families": list(original.SPLITS["dev"]),
        "projection_sha256": original.digest(value),
        "no_test_qualification": True,
        "no_gpu_run": True,
        "no_native_orcel_execution": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Never overwrite H25 data or a prior projection")
    value = candidate()
    receipt = validate(value)
    out.mkdir(parents=True, exist_ok=False)
    (out / "projection.json").write_bytes(original.canonical(value))
    (out / "manifest.json").write_bytes(original.canonical(receipt))
    (out / "model-inputs.jsonl").write_bytes(b"".join(original.canonical(x["model_input"]) for x in value["rows"]))
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
