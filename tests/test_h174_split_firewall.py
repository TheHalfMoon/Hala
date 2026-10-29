"""H174 strictly metadata-only split provenance negative tests."""
from copy import deepcopy
import pytest
from h174.split_firewall import SCHEMA, HISTORY_SCHEMA, validate


def sample():
    return ({
        "schema": SCHEMA,
        "sources": [
            {"app": "newclick", "split": "TRAIN", "repo": "example/train-ui",
             "git_sha": "1"*40, "html_sha256": "1"*64, "license": "MIT"},
            {"app": "newdrag", "split": "DEV", "repo": "example/dev-ui",
             "git_sha": "2"*40, "html_sha256": "2"*64, "license": "MIT"},
            {"app": "newselect", "split": "OOD_RESERVED", "repo": "example/ood-ui",
             "git_sha": "3"*40, "html_sha256": "3"*64, "license": "MIT"},
        ],
        "cases": [
            {"index": 1, "app": "newclick", "split": "TRAIN",
             "image_sha256": "4"*64, "visual_rgba_sha256": "5"*64,
             "language": "en", "action_type": "click",
             "instruction": "Click the correct button"},
            {"index": 2, "app": "newdrag", "split": "DEV",
             "image_sha256": "6"*64, "visual_rgba_sha256": "7"*64,
             "language": "ar", "action_type": "drag",
             "instruction": "اسحب العنصر إلى الهدف"},
        ],
    }, {"schema": HISTORY_SCHEMA, "image_shas": ["8"*64],
        "visual_shas": ["9"*64]})


def test_fresh_source_and_images_pass_metadata_only():
    data, history = sample()
    result = validate(data, history)
    assert result["train_cases"] == result["dev_cases"] == 1
    assert result["reserved_ood_app_count"] == 1
    assert result["model_calls"] == result["native_rewards"] == 0


@pytest.mark.parametrize("historical_app", [
    "tingle", "tiny-slider", "flatpickr", "awesomplete",
    "pell", "pikaday", "choices", "nouislider",
])
def test_cannot_use_historical_train_or_consumed_eval_app(historical_app):
    data, history = sample()
    data["sources"][0]["app"] = historical_app
    data["cases"][0]["app"] = historical_app
    with pytest.raises(ValueError, match="historical source app"):
        validate(data, history)


def test_ood_reserved_source_never_has_visible_scenes():
    data, history = sample()
    case = deepcopy(data["cases"][0])
    case.update(index=3, app="newselect", split="OOD_RESERVED",
                image_sha256="a"*64, visual_rgba_sha256="b"*64)
    data["cases"].append(case)
    with pytest.raises(ValueError, match="no OOD images"):
        validate(data, history)


@pytest.mark.parametrize("field,value", [
    ("reference_point", [45, 56]),
    ("owner_gold", {"reference_after_index": 2}),
    ("oracle", True),
    ("reference_coordinates", [45, 56]),
])
def test_reject_leaked_owner_gold_fields(field, value):
    data, history = sample()
    data["cases"][0][field] = value
    with pytest.raises(ValueError, match="gold, reference"):
        validate(data, history)


def test_same_source_cannot_appear_in_train_and_dev():
    data, history = sample()
    data["sources"][1]["app"] = "newclick"
    with pytest.raises(ValueError, match="multiple splits"):
        validate(data, history)


@pytest.mark.parametrize("source_index", [1, 2])
def test_different_commits_of_same_repo_cannot_cross_train_dev_or_ood(source_index):
    data, history = sample()
    data["sources"][source_index]["repo"] = "EXAMPLE/TRAIN-UI"
    with pytest.raises(ValueError, match="duplicate upstream repository"):
        validate(data, history)


@pytest.mark.parametrize("repo", ["../private", "example/../private", "example/ui/extra", "example\\ui", ""])
def test_invalid_upstream_repo_metadata_rejected(repo):
    data, history = sample()
    data["sources"][0]["repo"] = repo
    with pytest.raises(ValueError, match="upstream repository"):
        validate(data, history)


def test_missing_upstream_repo_is_not_valid_v2_metadata():
    data, history = sample()
    data["sources"][0].pop("repo")
    with pytest.raises(ValueError, match="allowed provenance"):
        validate(data, history)


def test_old_v1_metadata_contract_cannot_bypass_repository_identity():
    data, history = sample()
    data["schema"] = "hala.h174.new-original-app-train-dev-metadata-preflight.v1"
    with pytest.raises(ValueError, match="wrong candidate schema"):
        validate(data, history)


def test_unhashable_repo_metadata_fails_closed():
    data, history = sample()
    data["sources"][0]["repo"] = ["example/train-ui"]
    with pytest.raises(ValueError, match="upstream repository"):
        validate(data, history)


@pytest.mark.parametrize("field,history_key", [
    ("image_sha256", "image_shas"),
    ("visual_rgba_sha256", "visual_shas"),
])
def test_reject_historical_sha_overlap_without_opening_old_images(field, history_key):
    data, history = sample()
    history[history_key] = [data["cases"][0][field]]
    with pytest.raises(ValueError, match="historical or cross-case"):
        validate(data, history)


def test_reject_identical_visual_pixels_under_distinct_png_encoding():
    data, history = sample()
    data["cases"][1]["visual_rgba_sha256"] = data["cases"][0]["visual_rgba_sha256"]
    with pytest.raises(ValueError, match="historical or cross-case"):
        validate(data, history)


def test_reject_empty_or_unrecognized_historical_sha_registry():
    data, history = sample()
    history["image_shas"] = []
    with pytest.raises(ValueError, match="nonempty historical hashes"):
        validate(data, history)
    data, history = sample()
    history["reference_positions"] = [1]
    with pytest.raises(ValueError, match="owner historical SHA-only"):
        validate(data, history)


def test_no_ood_reserved_app_does_not_qualify_protocol():
    data, history = sample()
    data["sources"].pop()
    with pytest.raises(ValueError):
        validate(data, history)


def test_unknown_fields_and_invalid_hashes_fail_closed():
    data, history = sample()
    data["source_gold"] = []
    with pytest.raises(ValueError, match="unrecognized top-level"):
        validate(data, history)
    data, history = sample()
    data["cases"][0]["image_sha256"] = "notsha"
    with pytest.raises(ValueError, match="PNG/RGBA SHA"):
        validate(data, history)

@pytest.mark.parametrize("field,value", [
    ("app", ["not-an-app"]), ("split", ["TRAIN"]),
])
def test_unhashable_user_metadata_must_fail_closed_as_value_error(field, value):
    data, history = sample()
    data["cases"][0][field] = value
    with pytest.raises(ValueError, match="H174 metadata firewall"):
        validate(data, history)


def test_unhashable_source_split_must_fail_closed_as_value_error():
    data, history = sample()
    data["sources"][0]["split"] = ["TRAIN"]
    with pytest.raises(ValueError, match="H174 metadata firewall"):
        validate(data, history)

def test_reject_renamed_historical_app_using_same_source_revision():
    from h173.source_preflight import SOURCE_PINS
    data, history = sample()
    data["sources"][0]["git_sha"] = SOURCE_PINS["tingle"]["git_sha"]
    with pytest.raises(ValueError, match="historical or aliased"):
        validate(data, history)

def test_reject_renamed_historical_app_using_same_html_source():
    from h173.source_preflight import SOURCE_PINS
    data, history = sample()
    data["sources"][0]["html_sha256"] = SOURCE_PINS["tingle"]["original_html_sha256"]
    with pytest.raises(ValueError, match="historical or aliased"):
        validate(data, history)


def test_two_new_names_cannot_share_same_git_or_html_source():
    for field in ("git_sha", "html_sha256"):
        data, history = sample()
        data["sources"][1][field] = data["sources"][0][field]
        with pytest.raises(ValueError, match="historical or aliased"):
            validate(data, history)
