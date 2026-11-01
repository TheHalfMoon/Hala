from __future__ import annotations
from fractions import Fraction
from math import comb

MAX_FAMILIES = 64  # Match H340's independent-family bound.

def familywise_paired_gate(rows: object) -> dict:
    # Invented-only family-level comparison. Does not load images or weights.
    if type(rows) is not tuple or not 6 <= len(rows) <= MAX_FAMILIES:
        raise ValueError('H345_REFUSED')
    seen = set()
    wins = losses = ties = 0
    for row in rows:
        if (type(row) is not tuple or len(row) != 3
                or type(row[0]) is not str or not row[0]
                or len(row[0]) > 48 or not row[0].isascii()
                or not all(c.isalnum() or c in '-_' for c in row[0])
                or row[0] in seen or type(row[1]) is not bool
                or type(row[2]) is not bool):
            raise ValueError('H345_REFUSED')
        seen.add(row[0])
        base, adapted = row[1:]
        wins += adapted and not base
        losses += base and not adapted
        ties += base == adapted
    n = wins + losses
    # Exact two-sided sign test across independent held-out families.
    p = min(Fraction(1), Fraction(2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)), 2**n)) if n else Fraction(1)
    return {
        'families': len(rows), 'wins': wins, 'losses': losses, 'ties': ties,
        'two_sided_exact_sign_p': float(p),
        'two_sided_exact_sign_ratio': {'numerator': p.numerator, 'denominator': p.denominator},
        'research_only': True, 'training_admitted': False,
        'blind_ood_verified': False, 'model_gain_claim_allowed': False,
        'release_allowed': False,
    }
