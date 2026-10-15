"""Task-level metrics with honest dependence and synthetic-only scope."""
import math


def wilson(successes, count, z=1.959963984540054):
    if type(count) is not int or type(successes) is not int or not 0 <= successes <= count:
        raise ValueError("Invalid binomial sample counts")
    if count == 0:
        return [None, None]
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def _stratum(results):
    count = len(results)
    successes = sum(r["verified_success"] for r in results)
    return {"sample_count": count, "verified_success_count": successes, "verified_success_rate": successes / count if count else None, "task_wilson_95": wilson(successes, count),
            "invalid_outputs": sum(r["invalid_outputs"] for r in results), "wrong_target_count": sum(r["wrong_target"] for r in results),
            "schema_failures": sum(r["schema_failures"] for r in results), "stale_failures": sum(r["stale_failures"] for r in results), "false_done_count": sum(r["false_done"] for r in results)}


def summarize(results):
    summary = _stratum(results)
    families = {}
    for result in results:
        families.setdefault(result["family_id"], []).append(result)
    fully_passed = sum(all(r["verified_success"] for r in cases) for cases in families.values())
    summary.update(claim="synthetic infrastructure baseline only; no real model quality measurement", independent_family_count=len(families), fully_passed_family_count=fully_passed,
                   family_wilson_95=wilson(fully_passed, len(families)), uncertainty_note="Descriptive binomial intervals only. Tasks share workflow families; variants are not independent families. Neither interval estimates real model performance.",
                   language_strata={language: _stratum([r for r in results if r["language"] == language]) for language in sorted({r["language"] for r in results})},
                   suite_strata={suite: _stratum([r for r in results if r["suite"] == suite]) for suite in sorted({r["suite"] for r in results})})
    return summary
