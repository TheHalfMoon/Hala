"""H311: exact familywise hypothetical paired-GUI inference, not a score reader.

No private per-case or aggregate input, files, media, source prompts, model
outputs, OOD cases, or external services. Call only with INVENTED planning
counts. This cannot authenticate source rights, independence, or blindness.
"""
from __future__ import annotations

from fractions import Fraction

from h310.exact_discordant_study_design import (
    ALPHA, MAX_DISCORDANT, exact_two_sided_p,
)

MIN_FAMILIES = 2
MAX_FAMILIES = 8


class FamilywiseRefused(ValueError):
    """Fixed public diagnostic without echoing input."""


def _need(ok: bool) -> None:
    if not ok:
        raise FamilywiseRefused("H311_INVALID_HYPOTHETICAL_FAMILY_DESIGN")


def hypothetical_familywise_gate(
    families: tuple[tuple[str, int, int], ...],
    *,
    expected_family_count: int,
    alpha: Fraction = ALPHA,
) -> dict:
    """Exact two-sided McNemar plus Holm across ALL named hypothetical families.

    Each tuple is (invented family ID, BASE-only, ADAPTED-only). Holm's
    familywise error control does not require independence ACROSS valid
    per-family p-values. Exact conditional McNemar still requires valid
    independent paired sampling WITHIN a family, which is NOT established.
    No family can be omitted after seeing outcomes; the caller must freeze
    the complete set prospectively. This function cannot prove that freeze.
    """
    _need(type(expected_family_count) is int
          and MIN_FAMILIES <= expected_family_count <= MAX_FAMILIES)
    _need(type(families) is tuple
          and len(families) == expected_family_count)
    _need(type(alpha) is Fraction and Fraction(0) < alpha < Fraction(1))

    rows = []
    for i, row in enumerate(families, start=1):
        _need(type(row) is tuple and len(row) == 3)
        family, base_only, adapted_only = row
        _need(type(family) is str and family == f"family_{i}")
        _need(type(base_only) is int and type(adapted_only) is int)
        _need(base_only >= 0 and adapted_only >= 0
              and base_only + adapted_only <= MAX_DISCORDANT)
        rows.append({
            "family": family,
            "p": exact_two_sided_p(base_only, adapted_only),
            "positive": adapted_only > base_only,
        })

    # Stable tie ordering and step-down monotonic Holm adjusted p-values.
    ordered = sorted(range(len(rows)), key=lambda i: (rows[i]["p"], i))
    adjusted: list[Fraction] = [Fraction(1)] * len(rows)
    running = Fraction(0)
    for rank, index in enumerate(ordered):
        running = max(running, min(
            Fraction(1), (len(rows) - rank) * rows[index]["p"]
        ))
        adjusted[index] = running

    family_results = tuple({
        "family": row["family"],
        "two_sided_exact_p": str(row["p"]),
        "holm_adjusted_p": str(adjusted[i]),
        "positive_direction": row["positive"],
        "positive_and_holm_significant": (
            row["positive"] and adjusted[i] <= alpha
        ),
    } for i, row in enumerate(rows))

    return {
        "status": "H311_INVENTED_FAMILYWISE_DESIGN_ONLY",
        "expected_family_count": expected_family_count,
        "two_sided_familywise_alpha": str(alpha),
        "by_hypothetical_family": family_results,
        "all_families_positive_and_holm_significant_hypothetically": all(
            row["positive_and_holm_significant"] for row in family_results
        ),
        "preregistration_independently_authenticated": False,
        "independent_within_family_pairs_verified": False,
        "independent_custodian_or_blind_ood_verified": False,
        "source_rights_or_train_dev_admission_verified": False,
        "real_base_vs_adapted_gain_proven": False,
        "can_release": False,
    }
