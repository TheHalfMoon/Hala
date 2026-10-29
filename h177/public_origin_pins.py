"""H177: independently pinned old upstream repository identities.

H176 pairs owner-supplied names, repositories and source hashes, but those
pairs may still contain invented repository names. H177 independently binds
the historical sources whose original Git repository and exact revision are
already documented by Hala's earlier public source modules.

Only PUBLIC source identity metadata is embedded here. This does not open
earlier DEV/OOD images, labels, predictions, private manifests or adapters.
No completed audit of all historical families is claimed.
"""
from __future__ import annotations

from h176.source_family_attribution import audit_attribution

# Evidence anchors in this repo:
# h138/README.md: two evaluation-only original Kanban repositories;
# h140/freeze_train_dev.py: three previously used TRAIN/DEV original boards;
# h158/README.md and original_todomvc_source.py: original TodoMVC;
# h163/VERIFIED_RESULTS.md and source.py: former GridStack OOD;
# h164/PILOT_VERIFIED_RESULTS.md and four_app_train_pilot.py:
# original simple-table and calculator TRAIN sources.
# Spellings and Git commits are copied from those pinned source records.
# A repo identified in this catalog is NOT newly unseen, even at a
# different Git commit, under an alternate app name, or in OOD_RESERVED.
KNOWN_PUBLIC_PREVIOUS = {
    "react-dnd-kit-board": (
        "MSabriBoy/Kanban_Task_Board",
        "553d4e92be3e86e0deb7045f4968152868399eae",
    ),
    "react-html5-board": (
        "9bzero/kanban-board",
        "83413e1f5fab4690bb46b23201a3438a427ef265",
    ),
    "vanilla-native-board": (
        "anantagarwal1307/Drag-Drop-Kanban-Board",
        "47cfba359a71b70a6a23e511e554416aec006474",
    ),
    "react-native-board": (
        "dandemslay/React-KanbanBoard",
        "fa047f7a32afb57750ed5246456dc282ea7befcd",
    ),
    "react-dndkit-board": (
        "TomCani/kanban-dndkit-localstorage",
        "f3e129998e3140a386f1884faff314345e0eb10b",
    ),
    "todomvc-es6": (
        "tastejs/todomvc",
        "ff43b02e59dfa604386bb382034b2cd07c2bcd8a",
    ),
    "gridstack-original-column-demo": (
        "gridstack/gridstack.js",
        "b05b79dcbca21caca046c1e8c4da6e59546ec806",
    ),
    "simple-table": (
        "arce/simple-table",
        "7580270d18fb764bbf437a5e5b88ea66d63fc449",
    ),
    "calculator": (
        "ashishxa/calculator",
        "8f27a92072dde9658d8d94db8918a184ed7bd416",
    ),
}

# All verified exact source roots must remain blocked regardless of whether
# the owner separately supplies the matching historical registry.
KNOWN_PUBLIC_REPOS = frozenset(
    repo.casefold() for repo, _ in KNOWN_PUBLIC_PREVIOUS.values()
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("H177 public historical origin check: " + reason)


def audit_public_origin_pins(
    metadata: dict,
    owner_hash_registry: dict,
    source_history: dict,
    owner_attribution: dict,
) -> dict:
    """Require exact known historical app->upstream Git provenance.

    The H176 owner metadata crosscheck runs FIRST and remains mandatory.
    H177 then rejects falsified repository attribution for known past apps
    and an alternate-commit fresh source of any previously used repo.
    Historical HTML authenticity and complete past-family coverage remain
    strictly unproven, regardless of this result.
    """
    h176 = audit_attribution(
        metadata, owner_hash_registry, source_history, owner_attribution
    )
    # H176 enforces a strict owner attribution schema. Scan only the four
    # permitted source identity fields; never accept gold or media fields.
    by_app = {
        row["app"]: row
        for row in owner_attribution["families"]
    }
    missing = set(KNOWN_PUBLIC_PREVIOUS).difference(by_app)
    _require(not missing, "known prior TRAIN/DEV/OOD source attribution missing")

    for app, (repo, sha) in KNOWN_PUBLIC_PREVIOUS.items():
        row = by_app[app]
        _require(
            row.get("repo", "").casefold() == repo.casefold()
            and row.get("git_sha") == sha,
            "source-repository or Git pin contradicts prior original app",
        )

    # An owner-supplied historical list cannot be used to reclassify one of
    # these public previously used original repos as "new" at another SHA.
    for source in metadata["sources"]:
        _require(
            source["repo"].casefold() not in KNOWN_PUBLIC_REPOS,
            "candidate reuses publicly attested historical upstream repo",
        )

    return {
        "status": "H177_PUBLIC_PRIOR_ORIGINS_METADATA_ONLY_NOT_ADMISSION",
        "public_git_repository_pin_pairs_checked": len(KNOWN_PUBLIC_PREVIOUS),
        "known_public_previous_app_ids_at_least": len(by_app),
        "historical_repositories_compared": h176[
            "distinct_historical_repositories"
        ],
        "owner_attribution_fingerprint_sha256": h176["attribution_sha256"],
        "historical_source_completeness_proven": False,
        "old_html_source_authenticity_proven": False,
        "training_approved": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "native_rewards": 0,
    }
