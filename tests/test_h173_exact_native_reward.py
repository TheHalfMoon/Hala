"""Native GUI reward must match real requested outcome, not any movement."""
import pytest

from h173.capture_ood import exact_native_reward, original_carousel_direction_ok


def test_real_modal_closed_is_success_only_from_open_initial_state():
    prior = {"kind": "modal_visible", "value": True}
    assert exact_native_reward("tingle", prior, modal_visible=False) is True
    assert exact_native_reward("tingle", prior, modal_visible=True) is False


@pytest.mark.parametrize("prior", [
    {"kind": "modal_visible", "value": False},
    {"kind": "carousel_index", "value": 2},
])
def test_false_original_modal_prestate_never_scored(prior):
    with pytest.raises(ValueError):
        exact_native_reward("tingle", prior, modal_visible=False)


def test_carousel_must_arrive_at_owner_correct_post_index():
    prior = {"kind": "carousel_index", "value": 4}
    assert exact_native_reward(
        "tiny-slider", prior, observed_index=7, owner_after_index=7
    ) is True
    # An opposite-direction click has still moved the index, but is WRONG.
    assert exact_native_reward(
        "tiny-slider", prior, observed_index=3, owner_after_index=7
    ) is False
    assert exact_native_reward(
        "tiny-slider", prior, observed_index=4, owner_after_index=7
    ) is False


@pytest.mark.parametrize("initial, observed, wanted", [
    (4, 3, 4),  # Owner target identical to original is not valid reference
    (4, 7, None),
    (4, "7", 7),
    ("4", 7, 7),
])
def test_missing_or_forged_carousel_direction_target_rejected(initial, observed, wanted):
    with pytest.raises(ValueError):
        exact_native_reward(
            "tiny-slider",
            {"kind": "carousel_index", "value": initial},
            observed_index=observed,
            owner_after_index=wanted,
        )


def test_unsupported_original_app_never_scores():
    with pytest.raises(ValueError):
        exact_native_reward(
            "Pikaday", {"kind": "modal_visible", "value": True},
            modal_visible=False,
        )


@pytest.mark.parametrize("direction,after,valid", [
    ("prev", 3, True),
    ("prev", 5, False),
    ("next", 5, True),
    ("next", 3, False),
    ("prev", 4, False),
    ("next", 4, False),
    ("wrong", 5, False),
    ("next", True, False),
])
def test_original_carousel_reference_must_follow_requested_direction(direction, after, valid):
    prestate = {"kind": "carousel_index", "value": 4, "direction": direction}
    assert original_carousel_direction_ok(prestate, after) is valid


def test_original_carousel_reference_rejects_noninteger_initial_index():
    assert not original_carousel_direction_ok(
        {"kind": "carousel_index", "value": True, "direction": "next"}, 5
    )
