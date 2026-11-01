"""H310: prospective paired-GUI study-design arithmetic, never a score reader.

No model output, private receipt, per-case label, source screenshot, task text,
coordinate, identifier, manifest, or historical OOD evidence is accepted.
The user-selected probability is a HYPOTHETICAL scenario, not an estimate.
"""
from __future__ import annotations

import json
from fractions import Fraction
from math import comb

MAX_DISCORDANT = 128
ALPHA = Fraction(1, 20)
TARGET_POWER = Fraction(4, 5)
ILLUSTRATIVE_ADAPTED_WIN_PROB = Fraction(3, 4)


class DesignRefused(ValueError):
    """Fixed error text, without echoing any supplied value."""


def _require(condition: bool) -> None:
    if not condition:
        raise DesignRefused("H310_INVALID_PUBLIC_PLANNING_PARAMETER")


def _fraction(value: object, *, lower: Fraction, upper: Fraction) -> Fraction:
    _require(type(value) is Fraction and lower < value < upper)
    return value


def exact_two_sided_p(base_only: int, adapted_only: int) -> Fraction:
    """Conditional McNemar p for a known hypothetical 2x2 discordance pair."""
    _require(type(base_only) is int and type(adapted_only) is int)
    _require(base_only >= 0 and adapted_only >= 0)
    m = base_only + adapted_only
    _require(m <= MAX_DISCORDANT)
    if m == 0:
        return Fraction(1)
    k = min(base_only, adapted_only)
    return min(Fraction(1), Fraction(2 * sum(comb(m, i) for i in range(k + 1)), 2**m))


def conditional_positive_power(
    discordant_pairs: int, adapted_win_probability: Fraction,
    alpha: Fraction = ALPHA,
) -> Fraction:
    """Exact probability of positive-significant McNemar, GIVEN m discordants.

    This is NOT power as a function of the number of GUI tasks: the
    discordance rate, family dependence and task-level sampling are unknown.
    """
    _require(type(discordant_pairs) is int and 0 <= discordant_pairs <= MAX_DISCORDANT)
    q = _fraction(adapted_win_probability, lower=Fraction(1, 2), upper=Fraction(1))
    threshold = _fraction(alpha, lower=Fraction(0), upper=Fraction(1))
    m = discordant_pairs
    return sum((
        Fraction(comb(m, wins)) * q**wins * (1 - q)**(m - wins)
        for wins in range(m + 1)
        if wins > m - wins and exact_two_sided_p(m - wins, wins) <= threshold
    ), Fraction(0))


def prospective_discordance_grid(
    adapted_win_probability: Fraction = ILLUSTRATIVE_ADAPTED_WIN_PROB,
    alpha: Fraction = ALPHA,
    target_power: Fraction = TARGET_POWER,
    maximum: int = MAX_DISCORDANT,
) -> dict:
    """Fixed, PUBLIC hypothetical planning grid; no observed evidence."""
    q = _fraction(adapted_win_probability, lower=Fraction(1, 2), upper=Fraction(1))
    a = _fraction(alpha, lower=Fraction(0), upper=Fraction(1))
    target = _fraction(target_power, lower=Fraction(0), upper=Fraction(1))
    _require(type(maximum) is int and 1 <= maximum <= MAX_DISCORDANT)
    powers = [conditional_positive_power(m, q, a) for m in range(maximum + 1)]
    first = next((m for m in range(1, maximum + 1) if powers[m] >= target), None)
    # Exact binomial tests have discrete, nonmonotone power in m. Never
    # equate the FIRST qualifying m with a guarantee at the next m.
    stable = next((
        m for m in range(1, maximum + 1)
        if all(powers[k] >= target for k in range(m, maximum + 1))
    ), None)
    return {
        "status": "H310_HYPOTHETICAL_DISCORDANT_PAIRS_ONLY",
        "conditional_adapted_win_probability": str(q),
        "two_sided_alpha": str(a),
        "target_conditional_power": str(target),
        "maximum_discordant_pairs_searched": maximum,
        "first_discordant_count_meeting_target": first,
        "all_counts_from_here_meet_target_within_search_range": stable,
        "power_at_eight_discordant_pairs": str(powers[8]) if maximum >= 8 else None,
        "number_of_gui_tasks_required": None,
        "discordance_rate_known": False,
        "source_family_independence_verified": False,
        "private_holdout_constructed": False,
        "rights_or_official_train_dev_admission_verified": False,
        "observed_model_gain": False,
        "can_publish_or_release": False,
    }


def main() -> None:
    print(json.dumps(prospective_discordance_grid(), sort_keys=True))


if __name__ == "__main__":
    main()
