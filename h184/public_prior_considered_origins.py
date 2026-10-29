"""H184 public prior-original GUI repo exposure ledger.

These exact repository/Git identities appear in immutable Hala PUBLIC
historical source documents. Sources described as "considered" or "rejected"
are NOT represented as successful model training or scored evaluation.
Conservative novelty excludes *all* previously researched families,
including candidates and pilot/rejected apps, from a NEW unseen source.

No private owner source inventory, image SHA, predictions, gold or media.
This ledger is deliberately a lower bound, NOT historical completeness.
"""
from __future__ import annotations

import re

# app -> (original owner/repository, pinned original source Git SHA,
#         original Hala public source attestation, evidence class).
# H139 README and source/registry: two consumed TRAIN/DEV original boards
# and one consumed BLIND original board.
# H140 SOURCE_CANDIDATES + blind_source_preflight: proposed/rejected original
# boards, some inspected via owner native QA; do not claim all were scored.
# H154 VERIFIED_RESULTS + miniwob_source_probe: source-only native pilot.
PUBLIC_PREVIOUS_OR_CONSIDERED = {
    "h139-js-board": (
        "CodeWithSindhu/kanban-board-js",
        "82325355d168a9f4dba9f683899a0a6991034111",
        "h139/README.md", "past-train-dev",
    ),
    "h139-alt-static-board": (
        "arjundroid12/todo-drag-drop",
        "a4426599cb3778c8ee3d7c6b1fd27eed76bf5fe9",
        "h139/README.md", "past-train-dev",
    ),
    "h139-ood-board": (
        "CodeWithGionatha-Labs/dnd-kanban-board",
        "7ce11ad463b01a7b12b105417453d5251eabcd51",
        "h139/README.md", "past-blind",
    ),
    "h140-vue-blind-candidate": (
        "N3XT3R1337/kanban-board",
        "5d5a433d9860cec617ab523e91447fb6513ecee8",
        "h140/SOURCE_CANDIDATES.md", "proposed",
    ),
    "h140-aisha-blind-pilot": (
        "AishaChunawala/interactive-kanban-board-or-study-kanban-board",
        "f89d065e2cf16f91918bf1a72ae9ac132150197e",
        "h140/blind_source_preflight.py", "source-pilot",
    ),
    "h140-facetask-blind-candidate": (
        "aashwinshukla/FaceTask",
        "0698d3c0a2a4a63b636e40c62a21f645290a4705",
        "h140/SOURCE_CANDIDATES.md", "proposed",
    ),
    "h140-pixelpenguin-rejected": (
        "PixelPenguin94/kanban-board-vanilla-js",
        "652d763c61f59a23fbd64179e82bd363cda337e0",
        "h140/SOURCE_CANDIDATES.md", "rejected",
    ),
    "h140-michaelgermini-blind-pilot": (
        "michaelgermini/Kanban-Board",
        "a68c1d9d6281a793835ceaa5e4f620068d95200a",
        "h140/blind_source_preflight.py", "source-pilot",
    ),
    "h154-miniwob-original": (
        "Farama-Foundation/miniwob-plusplus",
        "7fd85d71a4b60325c6585396ec4f48377d049838",
        "h154/VERIFIED_RESULTS.md", "source-pilot",
    ),
}

REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
CLASS = frozenset({"past-train-dev", "past-blind",
                   "source-pilot", "proposed", "rejected"})

# Frozen public original source families are not prospective TRAIN/DEV/OOD,
# even if the original source checkout has been deleted from owner Windows.
BLOCKED_REPOSITORIES = frozenset(
    record[0].casefold() for record in PUBLIC_PREVIOUS_OR_CONSIDERED.values()
)
BLOCKED_COMMITS = frozenset(
    record[1] for record in PUBLIC_PREVIOUS_OR_CONSIDERED.values()
)


def validate_public_previous_ledger() -> dict:
    """Static public data shape guard: cannot establish actual completeness."""
    pairs = set()
    names = set()
    for name, row in PUBLIC_PREVIOUS_OR_CONSIDERED.items():
        if (
            type(name) is not str or type(row) is not tuple
            or len(row) != 4
        ):
            raise ValueError("H184 static public original ledger malformed")
        repo, sha, evidence, kind = row
        if (
            type(repo) is not str or REPO.fullmatch(repo) is None
            or any(p in (".", "..") for p in repo.split("/"))
            or type(sha) is not str or SHA.fullmatch(sha) is None
            or type(evidence) is not str or not evidence.startswith(
                ("h139/", "h140/", "h154/")
            )
            or not evidence.endswith((".md", ".py"))
            or kind not in CLASS
        ):
            raise ValueError("H184 original public prior pin is invalid")
        if repo.casefold() in names or sha in pairs:
            raise ValueError("H184 duplicate prior original source family")
        names.add(repo.casefold())
        pairs.add(sha)
    if len(pairs) != 9:
        raise ValueError("H184 public history ledger missing a pinned row")
    return {
        "status": "H184_PUBLIC_PRIOR_ORIGINS_ONLY_NOT_HISTORY_COMPLETE",
        "public_extra_repositories": len(names),
        "public_extra_git_pins": len(pairs),
        "complete_history_proven": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
    }
