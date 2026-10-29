"""Only synthetic H174 prior-source history metadata. Never reads prior pixels."""
from copy import deepcopy

import pytest
from h174.source_history_gate import (
    SCHEMA, validate_with_source_history,
)
from h174.split_firewall import (
    BANNED, BANNED_GIT, BANNED_HTML,
    SCHEMA as METADATA_SCHEMA, HISTORY_SCHEMA,
)


def fixture():
    meta = {
        "schema": METADATA_SCHEMA,
        "sources": [
            {"app": "freshclick", "split": "TRAIN", "repo": "newowner/click-demo",
             "git_sha": "1"*40, "html_sha256": "1"*64, "license": "MIT"},
            {"app": "freshdrag", "split": "DEV", "repo": "newowner/drag-demo",
             "git_sha": "2"*40, "html_sha256": "2"*64, "license": "MIT"},
            {"app": "freshood", "split": "OOD_RESERVED", "repo": "newowner/reserved-demo",
             "git_sha": "3"*40, "html_sha256": "3"*64, "license": "MIT"},
        ],
        "cases": [
            {"index": 1, "app": "freshclick", "split": "TRAIN",
             "image_sha256": "4"*64, "visual_rgba_sha256": "5"*64,
             "language": "en", "action_type": "click",
             "instruction": "Click the button here"},
            {"index": 2, "app": "freshdrag", "split": "DEV",
             "image_sha256": "6"*64, "visual_rgba_sha256": "7"*64,
             "language": "ar", "action_type": "drag",
             "instruction": "اسحب العنصر إلى الهدف"},
        ],
    }
    old_pixels = {
        "schema": HISTORY_SCHEMA, "image_shas": ["8"*64],
        "visual_shas": ["9"*64],
    }
    old_sources = {
        "schema": SCHEMA,
        "historical_app_ids": sorted(BANNED),
        "historical_repositories": ["history/other-donor"],
        "source_git_shas": sorted(BANNED_GIT),
        "source_html_shas": sorted(BANNED_HTML),
    }
    return meta, old_pixels, old_sources


def test_new_source_history_crosscheck_is_not_dataset_admission():
    meta, pixels, sources = fixture()
    receipt = validate_with_source_history(meta, pixels, sources)
    assert receipt["historical_repositories_compared"] == 1
    assert receipt["source_history_completeness_proven"] is False
    assert receipt["original_app_native_qa_proven"] is False
    assert receipt["training_approved"] is False
    assert receipt["model_calls"] == receipt["native_rewards"] == 0
    assert len(receipt["owner_source_registry_fingerprint_sha256"]) == 64


@pytest.mark.parametrize("index", [0, 1, 2])
def test_renamed_prior_repo_different_revision_rejected_in_any_split(index):
    meta, pixels, sources = fixture()
    # Neither the new name, git_sha nor original HTML SHA is historically known.
    # The same upstream repository remains historically burned regardless.
    meta["sources"][index]["repo"] = "HISTORY/OTHER-DONOR"
    with pytest.raises(ValueError, match="historical upstream repository family"):
        validate_with_source_history(meta, pixels, sources)


def test_prior_repository_alias_from_different_commit_not_approved():
    meta, pixels, sources = fixture()
    sources["historical_repositories"].append("NEWOWNER/CLICK-DEMO")
    with pytest.raises(ValueError, match="historical upstream repository family"):
        validate_with_source_history(meta, pixels, sources)


def test_owner_source_history_app_alias_detected():
    meta, pixels, sources = fixture()
    sources["historical_app_ids"].append("freshclick")
    with pytest.raises(ValueError, match="historical source app name"):
        validate_with_source_history(meta, pixels, sources)


def test_owner_only_git_sha_registry_detects_historical_source():
    meta, pixels, sources = fixture()
    sources["source_git_shas"].append(meta["sources"][0]["git_sha"])
    with pytest.raises(ValueError, match="historical original source Git SHA"):
        validate_with_source_history(meta, pixels, sources)


def test_owner_only_original_html_sha_registry_detects_reencoding_alias():
    meta, pixels, sources = fixture()
    sources["source_html_shas"].append(meta["sources"][1]["html_sha256"])
    with pytest.raises(ValueError, match="historical original source HTML"):
        validate_with_source_history(meta, pixels, sources)


@pytest.mark.parametrize("extra", [
    {"owner_gold": "private"}, {"images": ["raw"]}, {"reference_x": 5},
])
def test_source_registry_forbids_extra_gold_or_image_fields(extra):
    meta, pixels, sources = fixture()
    sources.update(extra)
    with pytest.raises(ValueError, match="exact SHA-only/source-identity fields"):
        validate_with_source_history(meta, pixels, sources)


def test_known_train_and_consumed_evaluation_names_mandatory():
    meta, pixels, sources = fixture()
    sources["historical_app_ids"].remove(next(iter(BANNED)))
    with pytest.raises(ValueError, match="known historical TRAIN/DEV/OOD"):
        validate_with_source_history(meta, pixels, sources)


def test_known_h173_git_and_html_pins_mandatory():
    for field in ("source_git_shas", "source_html_shas"):
        meta, pixels, sources = fixture()
        sources[field].clear()
        with pytest.raises(ValueError):
            validate_with_source_history(meta, pixels, sources)


@pytest.mark.parametrize("invalid", [
    ["example/new", "EXAMPLE/NEW"], ["example/new", "example/new"],
])
def test_duplicate_or_case_variant_historical_repository_rejected(invalid):
    meta, pixels, sources = fixture()
    sources["historical_repositories"] = invalid
    with pytest.raises(ValueError, match="duplicate|case-variant"):
        validate_with_source_history(meta, pixels, sources)


@pytest.mark.parametrize("invalid", [
    ["../outside"], ["a/b/extra"], [["owner", "name"]], [1],
])
def test_malformed_or_unhashable_historical_repository_fails_closed(invalid):
    meta, pixels, sources = fixture()
    sources["historical_repositories"] = invalid
    with pytest.raises(ValueError, match="repositories|repository"):
        validate_with_source_history(meta, pixels, sources)


def test_reject_empty_historical_repository_registry():
    meta, pixels, sources = fixture()
    sources["historical_repositories"] = []
    with pytest.raises(ValueError, match="bounded array"):
        validate_with_source_history(meta, pixels, sources)


def test_case_permutation_does_not_change_owner_history_fingerprint():
    meta, pixels, sources = fixture()
    sources["historical_repositories"].append("history/second-donor")
    first = validate_with_source_history(meta, pixels, sources)
    sources["historical_repositories"].reverse()
    second = validate_with_source_history(meta, pixels, sources)
    assert first["owner_source_registry_fingerprint_sha256"] == second[
        "owner_source_registry_fingerprint_sha256"
    ]


def test_history_crosscheck_cannot_override_core_split_firewall():
    meta, pixels, sources = fixture()
    meta["cases"][0]["owner_gold"] = [1, 2]
    with pytest.raises(ValueError, match="gold, reference"):
        validate_with_source_history(meta, pixels, sources)


def test_cannot_add_any_image_from_reserved_ood_family():
    meta, pixels, sources = fixture()
    case = deepcopy(meta["cases"][0])
    case.update(index=4, split="OOD_RESERVED", app="freshood",
                image_sha256="a"*64, visual_rgba_sha256="b"*64)
    meta["cases"].append(case)
    with pytest.raises(ValueError, match="no OOD images"):
        validate_with_source_history(meta, pixels, sources)
