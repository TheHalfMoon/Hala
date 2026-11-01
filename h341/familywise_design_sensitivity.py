"""H341: prospective family-level sign-test power, not model evidence.

Pure exact mathematics on invented win/tie probabilities. No model,
data admission, private outputs, legal rights or blind OOD.
"""
from __future__ import annotations
from fractions import Fraction
from math import comb
from h340.familywise_paired_inference import _two_sided_family_sign_p

ROLE = "H341_PROSPECTIVE_FAMILY_POWER_BOUNDS_ONLY_NOT_MODEL_EVIDENCE"
ALPHA = Fraction(1, 20)
MAX_FAMILIES = 64
CURRENT_PUBLIC_TRAIN_CANDIDATE_FAMILIES = 2
APPROVED_TRAIN_FAMILIES = 0


def _need(ok: bool) -> None:
    if not ok:
        raise ValueError("H341_INVALID_PROSPECTIVE_DESIGN")


def _fraction(x: Fraction) -> dict[str, int]:
    return {"numerator": x.numerator, "denominator": x.denominator}


def positive_raw_sign_rejection_probability(
    n_families: int,
    win_given_non_tie: Fraction,
    tie_probability: Fraction,
    raw_p_threshold: Fraction,
) -> Fraction:
    """Exact marginal power under hypothetical independent source families."""
    _need(type(n_families) is int and 1 <= n_families <= MAX_FAMILIES)
    _need(type(win_given_non_tie) is Fraction
          and 0 <= win_given_non_tie <= 1)
    _need(type(tie_probability) is Fraction
          and 0 <= tie_probability <= 1)
    _need(type(raw_p_threshold) is Fraction
          and raw_p_threshold in (ALPHA / 2, ALPHA))
    total = Fraction(0)
    for non_ties in range(n_families + 1):
        p_non_ties = (comb(n_families, non_ties)
                      * (1 - tie_probability) ** non_ties
                      * tie_probability ** (n_families - non_ties))
        if p_non_ties == 0:
            continue
        for wins in range(non_ties + 1):
            losses = non_ties - wins
            if wins <= losses:
                continue
            if _two_sided_family_sign_p(wins, losses) > raw_p_threshold:
                continue
            total += (p_non_ties * comb(non_ties, wins)
                      * win_given_non_tie ** wins
                      * (1 - win_given_non_tie) ** losses)
    _need(0 <= total <= 1)
    return total


def minimum_all_win_families_for_two_language_holm() -> int:
    """All-win families needed for both two-sided Holm endpoints."""
    for n in range(1, MAX_FAMILIES + 1):
        if 2 * _two_sided_family_sign_p(n, 0) <= ALPHA:
            return n
    raise AssertionError("unreachable")


def design_sensitivity(
    n_families: int,
    en_win_given_non_tie: Fraction,
    ar_win_given_non_tie: Fraction,
    tie_probability: Fraction,
) -> dict:
    """Conservative joint Holm power bounds with unknown EN/AR dependence.

    Both positive raw p <= .025 suffice; both <= .05 and at least one
    <= .025 are necessary. Frechet bounds avoid inventing language
    independence. All inputs are hypothetical, never observed model data.
    """
    _need(type(n_families) is int and 1 <= n_families <= MAX_FAMILIES)
    marginals = {}
    for lang, p in (("en", en_win_given_non_tie),
                    ("ar", ar_win_given_non_tie)):
        strict = positive_raw_sign_rejection_probability(
            n_families, p, tie_probability, ALPHA / 2)
        loose = positive_raw_sign_rejection_probability(
            n_families, p, tie_probability, ALPHA)
        _need(strict <= loose)
        marginals[lang] = (strict, loose)
    lower = max(Fraction(0), marginals["en"][0] + marginals["ar"][0] - 1)
    upper = min(Fraction(1), marginals["en"][1],
                marginals["ar"][1],
                marginals["en"][0] + marginals["ar"][0])
    _need(lower <= upper)
    return {
        "status": ROLE,
        "n_hypothetical_independent_families": n_families,
        "assumed_win_given_non_tie": {
            "en": _fraction(en_win_given_non_tie),
            "ar": _fraction(ar_win_given_non_tie),
        },
        "assumed_tie_probability": _fraction(tie_probability),
        "positive_raw_p_at_most_0_025_probability": {
            lang: _fraction(marginals[lang][0]) for lang in ("en", "ar")
        },
        "positive_raw_p_at_most_0_05_probability": {
            lang: _fraction(marginals[lang][1]) for lang in ("en", "ar")
        },
        "unknown_language_correlation_joint_holm_power_lower": _fraction(lower),
        "unknown_language_correlation_joint_holm_power_upper": _fraction(upper),
        "minimum_all_win_independent_families_for_both_languages":
            minimum_all_win_families_for_two_language_holm(),
        "current_public_source_train_candidate_families":
            CURRENT_PUBLIC_TRAIN_CANDIDATE_FAMILIES,
        "independently_approved_train_families": APPROVED_TRAIN_FAMILIES,
        "source_family_independence_attested": False,
        "legal_training_rights_approved": False,
        "blind_ood_custody_qualified": False,
        "genuine_base_adapted_model_gain_proven": False,
        "can_train_or_release": False,
    }
