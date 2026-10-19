"""Generate H28 source-only 7×2 counterbalanced public DEV plan.

No model calls, GPU allocation, source mutation, native actions or hidden TEST.
The separate evaluator file MUST NOT be added to model input messages.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from h28 import corpus, protocol


def json_lines(rows):
    return b"".join(corpus.canonical(row) for row in rows)


def prepare():
    dataset = corpus.dataset()
    source = corpus.validate(dataset)
    prompts, evaluator = [], []
    for row in dataset["records"]:
        for rotation in range(7):
            for instruction_locale in ("en", "ar"):
                ref = row["source_ref"]
                key = hashlib.sha256(
                    f'{source["sha256"]}|{ref["id"]}|{rotation}|{instruction_locale}'.encode("utf-8")
                ).hexdigest()[:32]
                prompts.append({
                    "request_key": key,
                    "messages": protocol.model_messages(row, rotation, instruction_locale),
                })
                evaluator.append({
                    "request_key": key,
                    "source_ref": ref,
                    "rotation": rotation,
                    "instruction_locale": instruction_locale,
                    "expected_wire": protocol.inverse_code(
                        row["private_reference"]["canonical_code"], rotation
                    ),
                })
    if len(prompts) != 896 or len(evaluator) != 896:
        raise ValueError("Incomplete counterbalanced H28 plan")
    if len({r["request_key"] for r in prompts}) != len(prompts):
        raise ValueError("Duplicate opaque request keys")
    if [r["request_key"] for r in prompts] != [r["request_key"] for r in evaluator]:
        raise ValueError("Model and evaluator alignment mismatch")
    by_rotation = Counter(x["rotation"] for x in evaluator)
    by_instruction = Counter(x["instruction_locale"] for x in evaluator)
    if by_rotation != {i: 128 for i in range(7)} or by_instruction != {"en": 448, "ar": 448}:
        raise ValueError("Unbalanced prospective experiment")
    # This source is not a blind TEST. No runtime or model has been invoked.
    prompt_bytes, evaluator_bytes = json_lines(prompts), json_lines(evaluator)
    manifest = {
        "schema": "hala.h28.prospective_dev_manifest.v1",
        "source_sha256": source["sha256"],
        "public_dev_only": True,
        "independent_holdout": False,
        "independent_bilingual_oracles": False,
        "real_visual_media": False,
        "model_inferences_run": 0,
        "gpu_calls": 0,
        "native_operations": 0,
        "cases": 64,
        "rotations": 7,
        "instruction_languages": ["en", "ar"],
        "planned_prompt_count": len(prompts),
        "model_messages_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "evaluator_sha256": hashlib.sha256(evaluator_bytes).hexdigest(),
        "warning": "The evaluator's private_reference is NOT a blind secret because source DEV is public; never pass evaluator file or metadata fields to model.",
    }
    return prompt_bytes, evaluator_bytes, manifest


def write_plan(output_directory):
    prompt_bytes, evaluator_bytes, manifest = prepare()
    destination = Path(output_directory).resolve()
    destination.mkdir(mode=0o700, exist_ok=False)
    # Two files are intentionally separate: passing oracle data as model context invalidates the evaluation.
    for name, contents in (
        ("model_messages.jsonl", prompt_bytes),
        ("evaluator_only.jsonl", evaluator_bytes),
        ("manifest.json", corpus.canonical(manifest)),
    ):
        with (destination / name).open("xb") as out:
            out.write(contents)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-directory", required=True, help="NEW destination; never overwrite")
    args = p.parse_args()
    print(json.dumps(write_plan(args.output_directory), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
