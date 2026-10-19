"""Create the NEW H28 supplementary paired-evidence public DEV source plan.

No model, cloud API, training, GPU or executor interaction. The evaluator-only
file is separate but NOT a blinded secret (the source code is public).
"""
import argparse
import hashlib
import json
from pathlib import Path

from h28 import contrasts

SCHEMA = "hala.h28.paired_evidence_plan.v1"


def prepare():
    data = contrasts.dataset()
    source = contrasts.validate(data)
    prompts, refs = [], []
    for item in data["records"]:
        ref = item["source_ref"]
        prompts.append({
            "request_key": ref["request_key"],
            "messages": contrasts.messages(item, "en"),
        })
        refs.append({
            "request_key": ref["request_key"],
            "source_ref": ref,
            "reference_code": item["private_reference"]["code"],
        })
    if len(prompts) != 72 or [r["request_key"] for r in prompts] != [
            r["request_key"] for r in refs]:
        raise ValueError("Unmatched prospective model/evaluator rows")
    messages = b"".join(contrasts.canonical(x) for x in prompts)
    evaluator = b"".join(contrasts.canonical(x) for x in refs)
    manifest = {
        "schema": SCHEMA,
        "source_sha256": source["source_sha256"],
        "source_public_dev": True,
        "blind_test": False,
        "independent_gold_review": False,
        "real_gpu_inference_count": 0,
        "paired_observations": 36,
        "planned_messages": 72,
        "model_messages_sha256": hashlib.sha256(messages).hexdigest(),
        "evaluator_only_sha256": hashlib.sha256(evaluator).hexdigest(),
        "warning": "Original cases and gold labels are public DEV; never send the evaluator sidecar or opaque request_key to the model.",
    }
    return messages, evaluator, manifest


def save(new_path):
    messages, evaluator, manifest = prepare()
    folder = Path(new_path).resolve()
    folder.mkdir(exist_ok=False)
    for name, content in (
        ("model_messages.jsonl", messages),
        ("evaluator_only.jsonl", evaluator),
        ("manifest.json", contrasts.canonical(manifest)),
    ):
        with (folder / name).open("xb") as file:
            file.write(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(save(args.output_directory), sort_keys=True))


if __name__ == "__main__":
    main()
