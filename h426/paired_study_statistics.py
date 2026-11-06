"""H426 matched-image evaluation statistics for honest research decisions.

A single screenshot with two prompts is ONE statistical sampling unit.
No model is loaded, no raw predictions or image source rows are returned.
This intentionally never certifies independent OOD, permissions or release.
"""
from __future__ import annotations

import hashlib
import math
import random

DOMAINS = frozenset(("slider", "slide_resize", "text_highlight", "sheet"))
MAX_IMAGES = 4096
BOOTSTRAPS = 1024
SEED = 4262026


def _checked(records):
    if (type(records) is not list or not 2 <= len(records) <= MAX_IMAGES):
        raise ValueError("H426 paired study requires 2..4096 distinct source images")
    seen = set()
    checked = []
    for row in records:
        if type(row) is not dict or set(row) != {
            "image_sha256", "domain", "source_family", "languages",
            "baseline_correct", "candidate_correct",
        }:
            raise ValueError("H426 paired study record shape changed")
        sha = row["image_sha256"]
        domain = row["domain"]
        family = row["source_family"]
        langs = row["languages"]
        if (type(sha) is not str or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
            or sha in seen
            or type(domain) is not str or domain not in DOMAINS
            or type(family) is not str or not 1 <= len(family) <= 160
            or type(langs) is not list or len(langs) != 2
            or any(x not in ("en", "ar") or type(x) is not str for x in langs)
            or type(row["baseline_correct"]) is not list
            or type(row["candidate_correct"]) is not list
            or len(row["baseline_correct"]) != 2
            or len(row["candidate_correct"]) != 2
            or any(type(x) is not bool for x in
                   row["baseline_correct"] + row["candidate_correct"])):
            raise ValueError("H426 paired study source or boolean outcome drift")
        seen.add(sha)
        checked.append((
            sha, domain, family, tuple(langs),
            tuple(row["baseline_correct"]), tuple(row["candidate_correct"]),
        ))
    # Stable output under input reordering; never print IDs or source families.
    return sorted(checked)


def _mcnemar_two_sided(better, worse):
    """Exact two-sided paired sign p for discordant both-correct images."""
    if better < 0 or worse < 0 or type(better) is not int or type(worse) is not int:
        raise ValueError("H426 invalid discordance counts")
    n = better + worse
    if not n:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2.0 * (sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))


def _bootstrap_interval(scores):
    """Cluster bootstrap over screenshot-level action-count differences."""
    rng = random.Random(SEED)
    n = len(scores)
    distribution = []
    for _ in range(BOOTSTRAPS):
        distribution.append(
            sum(scores[rng.randrange(n)] for _ in range(n)) / (2 * n)
        )
    distribution.sort()
    lo = distribution[int((BOOTSTRAPS - 1) * 0.025)]
    hi = distribution[int((BOOTSTRAPS - 1) * 0.975)]
    return [round(lo, 6), round(hi, 6)]


def summarize_matched_images(records):
    """Use one matched source PNG/JPEG as one resampling/discordance unit.

    These aggregate statistics are never evidence of blinded custody by
    themselves; no source-rights or model-provenance attestation is verified.
    """
    rows = _checked(records)
    n = len(rows)
    baseline_actions = candidate_actions = 0
    old_both = new_both = 0
    improved = regressed = unchanged = 0
    b_only = c_only = 0
    deltas = []
    domain_stats = {d: {"images": 0, "baseline_correct_actions": 0,
                        "candidate_correct_actions": 0,
                        "baseline_both_correct": 0,
                        "candidate_both_correct": 0}
                    for d in sorted(DOMAINS)}
    for _, domain, _, langs, before, after in rows:
        original_count = sum(before)
        candidate_count = sum(after)
        baseline_actions += original_count
        candidate_actions += candidate_count
        old_both += int(original_count == 2)
        new_both += int(candidate_count == 2)
        delta = candidate_count - original_count
        deltas.append(delta)
        improved += int(delta > 0)
        regressed += int(delta < 0)
        unchanged += int(delta == 0)
        b_only += int(original_count == 2 and candidate_count != 2)
        c_only += int(original_count != 2 and candidate_count == 2)
        stat = domain_stats[domain]
        stat["images"] += 1
        stat["baseline_correct_actions"] += original_count
        stat["candidate_correct_actions"] += candidate_count
        stat["baseline_both_correct"] += int(original_count == 2)
        stat["candidate_both_correct"] += int(candidate_count == 2)
    domains = {name: stat for name, stat in domain_stats.items() if stat["images"]}
    return {
        "schema": "hala.h426.matched-source-image-study.v1",
        "image_units": n,
        "paired_action_units": 2 * n,
        "baseline_correct_actions": baseline_actions,
        "candidate_correct_actions": candidate_actions,
        "baseline_both_correct_images": old_both,
        "candidate_both_correct_images": new_both,
        "paired_image_improved": improved,
        "paired_image_regressed": regressed,
        "paired_image_unchanged": unchanged,
        "baseline_only_both_correct": b_only,
        "candidate_only_both_correct": c_only,
        "exact_two_sided_mcnemar_p_both_correct": round(
            _mcnemar_two_sided(b_only, c_only), 10
        ),
        "action_accuracy_difference_candidate_minus_baseline": round(
            (candidate_actions - baseline_actions) / (2 * n), 6
        ),
        "image_cluster_bootstrap_95pct_difference": _bootstrap_interval(deltas),
        "by_domain": domains,
        "statistical_unit": "original_image_with_two_instructions",
        "model_predictions_verified_here": False,
        "independent_blind_external_ood_attested_here": False,
        "copyright_train_rights_verified_here": False,
        "model_or_release_qualified": False,
    }



def summarize_synthetic_dev_pairs(pairs):
    """Re-score matched invented DEV predictions using frozen H425 evaluator.

    Each input is one screenshot: {index, domain, baseline, candidate}.
    Predictions contain IDs and raw text, but only aggregate output escapes.
    This cannot qualify an independent external benchmark or original rights.
    """
    if type(pairs) is not list or not 2 <= len(pairs) <= MAX_IMAGES:
        raise ValueError("H426 DEV comparison requires 2..4096 matched image pairs")
    from h426.synthetic_pair_eval import compare_synthetic_dev
    counted = []
    for entry in pairs:
        if (type(entry) is not dict
            or set(entry) != {"index", "domain", "baseline", "candidate"}
            or type(entry["index"]) is not int
            or entry["index"] < 0
            or type(entry["domain"]) is not str
            or entry["domain"] not in DOMAINS):
            raise ValueError("H426 source DEV evaluation entry schema mismatch")
        score = compare_synthetic_dev(
            entry["index"], entry["domain"],
            entry["baseline"], entry["candidate"],
        )
        counted.append({
            "image_sha256": score["source_image_sha256"],
            "domain": entry["domain"],
            "source_family": score["source_family"],
            "languages": ["en", "ar"],
            "baseline_correct": [
                score["per_language"][lang]["old_correct"]
                for lang in ("en", "ar")
            ],
            "candidate_correct": [
                score["per_language"][lang]["new_correct"]
                for lang in ("en", "ar")
            ],
        })
    out = summarize_matched_images(counted)
    # The source is author-generated, evaluated on original H425 metric,
    # and may be seen by the developer. It is NOT independent blind OOD.
    out["source_evaluation_split"] = "firstparty_invented_dev"
    out["external_ood_status"] = "not_independent"
    return out
