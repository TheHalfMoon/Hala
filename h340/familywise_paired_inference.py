"""H340: exact family-level paired Qwen inference, not evaluation qualification.

Pure CPU mathematics on invented or separately authorized categorical counts.
No images, prompts, source gold, model predictions, coordinates, weights, OOD,
Kaggle or legal rights evidence are read. No blind/LoRA/release claims.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb

ROLE = "H340_FAMILY_PAIRED_STATISTICS_ONLY_NO_BLIND_OOD_ATTESTATION"
LANGUAGES = ("en", "ar")
CATEGORIES = frozenset({
    "both", "base_only", "adapted_only", "neither",
    "base_abstain", "adapted_abstain", "base_unsafe", "adapted_unsafe",
})
ALPHA = Fraction(1, 20)
MAX_FAMILIES = 64
MAX_PAIRS_PER_FAMILY_LANGUAGE = 256


def _need(ok: bool) -> None:
    if not ok:
        raise ValueError("H340_INVALID_OR_UNQUALIFIED_AGGREGATE")


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _read_counts(value: object) -> dict[str, int]:
    _need(type(value) is dict and set(value) == CATEGORIES)
    _need(all(type(v) is int and 0 <= v <= MAX_PAIRS_PER_FAMILY_LANGUAGE
              for v in value.values()))
    counts = dict(value)
    n = sum(counts[k] for k in ("both", "base_only", "adapted_only", "neither"))
    _need(1 <= n <= MAX_PAIRS_PER_FAMILY_LANGUAGE
          and counts["base_abstain"] <= counts["adapted_only"] + counts["neither"]
          and counts["adapted_abstain"] <= counts["base_only"] + counts["neither"]
          and counts["base_unsafe"] <= n and counts["adapted_unsafe"] <= n)
    return counts


def _two_sided_family_sign_p(wins: int, losses: int) -> Fraction:
    """Exact sign test: independent non-tied *source families* are the units."""
    n = wins + losses
    if n == 0:
        return Fraction(1)
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**n))


def analyze_paired_families(families: object) -> dict:
    """Compute a predeclared two-sided family sign test with Holm EN/AR.

    Input: list of family aggregates WITHOUT family IDs. Each has EN and AR
    paired 2x2 counts, plus abstention/unsafe counters for both arms.
    This cannot independently prove family identity, source separation,
    prospective registration, lawful training, blind custody or model pins.
    """
    _need(type(families) is list and 2 <= len(families) <= MAX_FAMILIES)
    sums = {lang: {key: 0 for key in CATEGORIES} for lang in LANGUAGES}
    signs = {lang: {"wins": 0, "losses": 0, "ties": 0} for lang in LANGUAGES}
    for family in families:
        _need(type(family) is dict and set(family) == set(LANGUAGES))
        for lang in LANGUAGES:
            counts = _read_counts(family[lang])
            for key in CATEGORIES:
                sums[lang][key] += counts[key]
            difference = counts["adapted_only"] - counts["base_only"]
            signs[lang]["wins" if difference > 0 else
                        "losses" if difference < 0 else "ties"] += 1

    raw_p = {
        lang: _two_sided_family_sign_p(signs[lang]["wins"],
                                       signs[lang]["losses"])
        for lang in LANGUAGES
    }
    # Holm step-down across exactly two predeclared language endpoints.
    adjusted: dict[str, Fraction] = {}
    previous = Fraction(0)
    for index, lang in enumerate(sorted(LANGUAGES, key=lambda l: (raw_p[l], l))):
        previous = min(Fraction(1), max(previous, (2 - index) * raw_p[lang]))
        adjusted[lang] = previous

    results = {}
    for lang in LANGUAGES:
        c = sums[lang]
        n = sum(c[k] for k in ("both", "base_only", "adapted_only", "neither"))
        results[lang] = {
            "source_families": len(families),
            "paired_scenes": n,
            "base_successes": c["both"] + c["base_only"],
            "adapted_successes": c["both"] + c["adapted_only"],
            "adapted_minus_base_success_rate": _fraction(Fraction(
                c["adapted_only"] - c["base_only"], n)),
            "family_wins": signs[lang]["wins"],
            "family_losses": signs[lang]["losses"],
            "family_ties": signs[lang]["ties"],
            "exact_two_sided_family_sign_p": _fraction(raw_p[lang]),
            "holm_adjusted_p": _fraction(adjusted[lang]),
            "holm_below_0_05": adjusted[lang] < ALPHA,
            "base_abstentions": c["base_abstain"],
            "adapted_abstentions": c["adapted_abstain"],
            "base_unsafe": c["base_unsafe"],
            "adapted_unsafe": c["adapted_unsafe"],
        }
    return {
        "status": ROLE,
        "analysis_unit": "independent_source_family_not_scene",
        "family_count": len(families),
        "language_endpoints": results,
        "all_language_holm_below_0_05": all(
            results[lang]["holm_below_0_05"] for lang in LANGUAGES),
        "independent_family_custody_verified": False,
        "rights_and_train_admission_verified": False,
        "new_never_exposed_blind_ood_verified": False,
        "genuine_pinned_base_and_adapted_models_verified": False,
        "safety_and_abstention_qualified": False,
        "learned_model_gain_proven": False,
        "can_release": False,
    }
