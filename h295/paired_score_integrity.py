"""H295: evidence-integrity audit for EXPOSED first-party paired GUI research.

No screenshots, case identifiers, private model outputs, coordinates or
raw reports are accepted. This computes descriptive differences and, only
when identifiable, the EXACT two-sided conditional McNemar p-value.
A p-value is not a license, blind OOD, admission or release decision.

Input data is self-asserted and MUST be independently authenticated
against the real private kernel/closed verifier before treating as genuine.
"""
from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path

SCHEMA = "H295_EXPOSED_PAIRED_NATIVE_RESEARCH_V1"
DATASETS = {"queueboard": 2, "routeboard": 6}
MAX_BYTES = 4096
MAX_SAMPLES = 128
TOP_KEYS = frozenset({"schema", "source_exposure", "groups"})
GROUP_KEYS = frozenset({"family", "sample_count", "base_success",
                        "adapted_success", "paired"})
PAIR_KEYS = frozenset({"both_success", "base_only",
                       "adapted_only", "both_fail"})


class AnalysisRefused(ValueError):
    """Fixed wording; never reflect untrusted input to errors."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise AnalysisRefused("H295 " + code)


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    data = {}
    for key, val in pairs:
        require(key not in data, "DUPLICATE_JSON_KEY")
        data[key] = val
    return data


def reject_nonstandard(_value: str) -> None:
    raise AnalysisRefused("NONSTANDARD_JSON_NUMBER")


def parse(raw: bytes) -> dict:
    require(type(raw) is bytes and 16 <= len(raw) <= MAX_BYTES,
            "BOUNDED_AGGREGATE_REQUIRED")
    try:
        doc = json.loads(raw.decode("utf-8", errors="strict"),
                         object_pairs_hook=unique_keys,
                         parse_constant=reject_nonstandard)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AnalysisRefused("MALFORMED_AGGREGATE") from None
    require(type(doc) is dict and doc.keys() == TOP_KEYS,
            "UNEXPECTED_TOP_LEVEL_FIELDS")
    require(type(doc["schema"]) is str and doc["schema"] == SCHEMA,
            "AGGREGATE_SCHEMA_MISMATCH")
    require(type(doc["source_exposure"]) is str
            and doc["source_exposure"] == "public_previously_exposed",
            "NO_BLIND_SOURCE_PROMOTION")
    return doc


def natural(value: object, limit: int, code: str) -> int:
    require(type(value) is int and 0 <= value <= limit, code)
    return value


def exact_two_sided_mc_nemar(base_only: int, adapted_only: int) -> float:
    """Exact conditional binomial on discordant pairs, NOT unpaired z-test."""
    n = base_only + adapted_only
    require(n <= MAX_SAMPLES, "TOO_MANY_DISCORDANT_CASES")
    if n == 0:
        return 1.0
    extreme = min(base_only, adapted_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(extreme + 1)) / (2 ** n))


def analyze(data: dict) -> dict:
    require(type(data) is dict and data.keys() == TOP_KEYS
            and data["schema"] == SCHEMA
            and data["source_exposure"] == "public_previously_exposed",
            "UNTRUSTED_OR_NONEXPOSED_RESEARCH")
    groups = data["groups"]
    require(type(groups) is list and 1 <= len(groups) <= 2,
            "INVALID_GROUP_COUNT")
    out = []
    seen: set[str] = set()
    for row in groups:
        require(type(row) is dict and row.keys() == GROUP_KEYS,
                "FORBIDDEN_RAW_OR_MISSING_GROUP_FIELD")
        family = row["family"]
        require(type(family) is str and family in DATASETS
                and family not in seen, "UNREGISTERED_OR_DUPLICATE_FAMILY")
        seen.add(family)
        n = natural(row["sample_count"], MAX_SAMPLES, "INVALID_CASE_COUNT")
        require(n == DATASETS[family], "CHANGED_EXPOSED_CASE_PROTOCOL")
        base = natural(row["base_success"], n, "INVALID_BASE_SUCCESS")
        adapted = natural(row["adapted_success"], n, "INVALID_ADAPTED_SUCCESS")
        discordant = row["paired"]
        p_value = None
        exact_identifiable = False
        if discordant is not None:
            require(type(discordant) is dict
                    and discordant.keys() == PAIR_KEYS,
                    "INVALID_OR_RAW_PAIRING")
            a = natural(discordant["both_success"], n, "INVALID_PAIRING")
            b = natural(discordant["base_only"], n, "INVALID_PAIRING")
            c = natural(discordant["adapted_only"], n, "INVALID_PAIRING")
            d = natural(discordant["both_fail"], n, "INVALID_PAIRING")
            require(a + b + c + d == n
                    and base == a + b and adapted == a + c,
                    "PAIRED_AND_MARGINAL_TOTALS_DISAGREE")
            p_value = exact_two_sided_mc_nemar(b, c)
            exact_identifiable = True
        elif base == adapted:
            # If marginal totals are equal, then base_only==adapted_only
            # for ANY possible case pairing. Thus exact conditional
            # two-sided McNemar p=1 regardless of unseen pairing.
            p_value = 1.0
            exact_identifiable = True

        delta = adapted - base
        out.append({
            "family": family,
            "n": n,
            "base_success": base,
            "adapted_success": adapted,
            "adapted_minus_base": delta,
            "exact_p_value_identifiable": exact_identifiable,
            "exact_two_sided_p": p_value,
            "direction": ("higher" if delta > 0 else
                          "lower" if delta < 0 else "equal"),
            "research_only_exposed_source": True,
            "model_gain_or_generalization_proven": False,
        })

    out.sort(key=lambda item: ("queueboard", "routeboard").index(item["family"]))
    return {
        "status": "H295_PUBLIC_EXPOSED_RESEARCH_STATISTICS_ONLY",
        "by_family": out,
        "total_case_count": sum(row["n"] for row in out),
        "total_base_success": sum(row["base_success"] for row in out),
        "total_adapted_success": sum(row["adapted_success"] for row in out),
        "total_descriptive_delta": sum(row["adapted_minus_base"] for row in out),
        "combined_p_value_calculated": False,
        "source_lineage_or_legal_rights_authenticated": False,
        "official_train_dev_admitted": False,
        "new_independent_blind_ood_proven": False,
        "statistically_validated_generalization": False,
        "can_publish_or_release": False,
    }


def analyze_bytes(raw: bytes) -> dict:
    return analyze(parse(raw))


def analyze_aggregate_file(path: Path) -> dict:
    require(type(path) is Path and path.name.endswith(".h295-public-score.json")
            and path.is_file() and not path.is_symlink(),
            "AGGREGATE_FILE_IDENTITY")
    try:
        require(path.stat().st_size <= MAX_BYTES, "AGGREGATE_TOO_LARGE")
        return analyze_bytes(path.read_bytes())
    except OSError:
        raise AnalysisRefused("AGGREGATE_UNREADABLE") from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Public exposed research only; no OOD/admission/release"
    )
    parser.add_argument("--aggregate", required=True, type=Path)
    args = parser.parse_args()
    try:
        safe = analyze_aggregate_file(args.aggregate)
    except AnalysisRefused as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(safe, sort_keys=True))


if __name__ == "__main__":
    main()
