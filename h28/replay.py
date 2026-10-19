"""H28 hash-bound CPU consistency auditor for externally provided raw codes.

Even a passing audit DOES NOT attest an actual model, GPU, independent Arabic
ground truth, real UI postconditions or held-out TEST. Never dispatch actions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from h24.mind_dev import strict_json
from h28 import corpus, protocol
from tools.build_h28_dev_plan import prepare

SCHEMA = "hala.h28.external_dev_predictions.v1"
REPORT_SCHEMA = "hala.h28.cpu_source_replay_unqualified.v1"


def audit(path, expected_sha256):
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Externally observed lowercase SHA256 required")
    file = Path(path)
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 3_000_000:
        raise ValueError("External bounded regular result file required")
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("External raw results SHA256 changed")
    record = strict_json(data.decode("utf-8"))
    if type(record) is not dict or set(record) != {
            "schema", "source_sha256", "model_messages_sha256", "model_id",
            "model_revision", "provenance_claim", "predictions"}:
        raise ValueError("Invalid raw evidence envelope")
    _, oracle_bytes, manifest = prepare()
    if (record["schema"] != SCHEMA
            or record["source_sha256"] != corpus.PINNED_SOURCE_SHA256
            or record["model_messages_sha256"] != manifest["model_messages_sha256"]
            or type(record["model_id"]) is not str or not 0 < len(record["model_id"]) <= 160
            or type(record["model_revision"]) is not str
            or not 0 < len(record["model_revision"]) <= 160
            or record["provenance_claim"] not in ("synthetic_fixture", "external_unattested")):
        raise ValueError("Invalid H28 model/source provenance claim")
    refs = [json.loads(x) for x in oracle_bytes.splitlines()]
    outputs = record["predictions"]
    if type(outputs) is not list or len(outputs) != 896:
        raise ValueError("Missing H28 counterbalanced decisions")
    codes = Counter()
    semantic = Counter()
    by_lang = Counter()
    by_instruction = Counter()
    by_rotation = Counter()
    by_state = Counter()
    joint = {}
    conditional = {}
    format_valid = 0
    correct = 0
    withheld = 0
    explicit_refusals_on_danger = 0
    unsafe_wrong = 0
    permission_accesslike = 0
    errors = 0
    for expected, observed in zip(refs, outputs):
        if type(observed) is not dict or set(observed) != {"request_key", "raw", "error_type"}:
            raise ValueError("Invalid result row structure")
        if observed["request_key"] != expected["request_key"]:
            raise ValueError("Missing/duplicated/misordered source request key")
        raw = observed["raw"]
        error = observed["error_type"]
        if raw is None:
            if type(error) is not str or not 0 < len(error) <= 100:
                raise ValueError("Missing model error classification")
            errors += 1
        elif type(raw) is not str or len(raw) > 200 or error is not None:
            raise ValueError("Invalid or contradictory raw code/error")
        state = expected["source_ref"]["state"]
        verdict = protocol.score(raw, state, expected["rotation"])
        rotation = expected["rotation"]
        key_lang = expected["source_ref"]["language"]
        instruction_lang = expected["instruction_locale"]
        decoded = protocol.codebook(rotation).get(raw) if type(raw) is str else None
        dimensions = (key_lang, state, rotation, instruction_lang)
        cell = joint.setdefault(dimensions, {
            "cases": 0, "valid_format": 0, "inference_errors": 0,
            "semantic_correct": 0, "wire_distribution": Counter(),
            "decoded_semantic_distribution": Counter(),
        })
        cell["cases"] += 1
        cell["valid_format"] += int(verdict["format_valid"])
        cell["inference_errors"] += int(raw is None)
        cell["semantic_correct"] += int(verdict["semantic_correct"])
        cell["wire_distribution"][raw if raw is not None else "INFERENCE_ERROR"] += 1
        cell["decoded_semantic_distribution"][decoded or "INVALID"] += 1
        cond = conditional.setdefault((state, rotation), {
            "cases": 0, "valid_code_denominator": 0,
            "wire_c1": 0, "semantic_c1": 0,
        })
        cond["cases"] += 1
        if verdict["format_valid"]:
            cond["valid_code_denominator"] += 1
            cond["wire_c1"] += int(raw == "C1")
            cond["semantic_c1"] += int(decoded == "C1")
        format_valid += int(verdict["format_valid"])
        correct += int(verdict["semantic_correct"])
        withheld += int(verdict["withheld"])
        canonical = protocol.codebook(expected["rotation"]).get(raw) if type(raw) is str else None
        codes[raw if raw is not None else "INFERENCE_ERROR"] += 1
        semantic[canonical if canonical is not None else "INVALID"] += 1
        by_lang[(key_lang, bool(verdict["semantic_correct"]))] += 1
        by_instruction[(expected["instruction_locale"], bool(verdict["semantic_correct"]))] += 1
        by_rotation[(rotation, bool(verdict["semantic_correct"]))] += 1
        by_state[(state, bool(verdict["semantic_correct"]))] += 1
        if state == "unsafe_user":
            explicit_refusals_on_danger += int(verdict["explicit_refusal"])
        unsafe_wrong += int(verdict["unsafe_wrong_proposal"])
        permission_accesslike += int(verdict["unverified_permission_accesslike_proposal"])
    if sum(codes.values()) != 896:
        raise ValueError("Incomplete raw code replay")
    return {
        "schema": REPORT_SCHEMA,
        "raw_sha256": expected_sha256,
        "source_sha256": corpus.PINNED_SOURCE_SHA256,
        "model_messages_sha256": manifest["model_messages_sha256"],
        "source_kind_asserted_only": record["provenance_claim"],
        "model_id_asserted_only": record["model_id"],
        "model_revision_asserted_only": record["model_revision"],
        "scenarios": 64, "attempted_matrix_cells": 896,
        "format_valid": format_valid, "semantic_correct": correct,
        "withheld": withheld, "inference_errors_claimed": errors,
        "explicit_refusal_on_unsafe_user": explicit_refusals_on_danger,
        "unsafe_wrong_proposals": unsafe_wrong,
        "unverified_permission_accesslike_proposals": permission_accesslike,
        "wire_code_distribution": dict(sorted(codes.items())),
        "semantic_code_distribution": dict(sorted(semantic.items())),
        "correct_by_task_language": {lang: {
            "cases": sum(by_lang[(lang, b)] for b in (False, True)),
            "correct": by_lang[(lang, True)]}
            for lang in ("en", "ar")},
        "correct_by_instruction_language": {lang: {
            "cases": sum(by_instruction[(lang, b)] for b in (False, True)),
            "correct": by_instruction[(lang, True)]}
            for lang in ("en", "ar")},
        "correct_by_rotation": {str(r): {
            "cases": sum(by_rotation[(r, b)] for b in (False, True)),
            "correct": by_rotation[(r, True)]}
            for r in range(7)},
        "correct_by_state": {state: {
            "cases": sum(by_state[(state, b)] for b in (False, True)),
            "correct": by_state[(state, True)]}
            for state in corpus.STATES},
        "joint_task_language_state_rotation_instruction_language": [
            {"task_language": key[0], "state": key[1], "rotation": key[2],
             "instruction_language": key[3], "cases": value["cases"],
             "valid_format": value["valid_format"],
             "inference_errors": value["inference_errors"],
             "semantic_correct": value["semantic_correct"],
             "wire_distribution": dict(sorted(value["wire_distribution"].items())),
             "decoded_semantic_distribution": dict(sorted(
                 value["decoded_semantic_distribution"].items()))}
            for key, value in sorted(joint.items())],
        "wire_vs_semantic_c1_by_reference_state_rotation": [
            {"state": key[0], "rotation": key[1], **value,
             "wire_c1_rate_over_valid_codes": (
                 value["wire_c1"] / value["valid_code_denominator"]
                 if value["valid_code_denominator"] else None),
             "semantic_c1_rate_over_valid_codes": (
                 value["semantic_c1"] / value["valid_code_denominator"]
                 if value["valid_code_denominator"] else None)}
            for key, value in sorted(conditional.items())],
        "independent_test": False,
        "external_hardware_or_model_attested": False,
        "causal_hypothesis_verified": False,
        "independent_bilingual_oracle_adjudication": False,
        "production_or_native_authority": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_result")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    receipt = audit(args.raw_result, args.expected_sha256)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(receipt, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
