"""H75 bounded, deterministic bilingual and source-family processor audits.

The original all-media SHA and H67 source rederivation gates remain untouched.
A bounded processor audit is an explicitly requested FIRST-STEP diagnostic
only; default zero keeps the original full-record processor audit.
"""
from __future__ import annotations

from collections import Counter


def select_processor_audit_rows(records: list[dict], limit: int) -> list[dict]:
    if type(limit) is not int or limit < 0 or limit > 64:
        raise ValueError("processor audit limit must be 0 (full) or 1..64")
    if not records:
        raise ValueError("nonempty source-bound TRAIN records required")
    if limit == 0 or limit >= len(records):
        return list(records)

    groups = {}
    for i, row in enumerate(records):
        if not isinstance(row, dict) or row.get("language") not in ("ar", "en"):
            raise ValueError("processor audit needs verified bilingual TRAIN rows")
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError("processor audit family identity missing")
        groups.setdefault((row["language"], family), []).append(i)
    if len(groups) > limit:
        raise ValueError("audit budget cannot represent every language/source-family group")

    # Always include one row from EACH original (language, family) group.
    selected = {members[0] for _, members in sorted(groups.items())}
    remaining = limit - len(selected)
    if remaining:
        candidates = [i for i in range(len(records)) if i not in selected]
        for j in range(remaining):
            idx = (2*j + 1) * len(candidates) // (2*remaining)
            selected.add(candidates[idx])
    return [records[i] for i in sorted(selected)]


def audit_summary(rows: list[dict], original_count: int) -> dict:
    return {"processor_audited_records": len(rows),
            "source_records_total": original_count,
            "audit_language_counts": dict(sorted(Counter(r["language"] for r in rows).items())),
            "audit_family_counts": dict(sorted(Counter(r["family"] for r in rows).items())),
            "audit_scope": ("full" if len(rows) == original_count
                            else "bounded_diagnostic_not_full_processor_audit")}
