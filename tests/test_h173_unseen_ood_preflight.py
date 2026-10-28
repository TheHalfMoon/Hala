"""H173 genuinely novel original source evidence vs OOD evaluation is separate."""
import pytest
from h173.source_preflight import (
    SOURCE_PINS,TRAIN_AND_CONSUMED_FAMILIES,OOD_COUNT,
    preregister_provisional_cases,assert_ood_not_qualified_without_new_receipts)

def test_two_new_source_families_disjoint_from_prior_original_apps():
    assert set(SOURCE_PINS)=={"tingle","tiny-slider"}
    assert set(SOURCE_PINS).isdisjoint(TRAIN_AND_CONSUMED_FAMILIES)
    assert len({x["git_sha"] for x in SOURCE_PINS.values()})==2
    assert len({x["family"] for x in SOURCE_PINS.values()})==2
    assert all(x["action"]=="click" for x in SOURCE_PINS.values())
    assert all(len(x["original_html_sha256"])==64 for x in SOURCE_PINS.values())

def test_ood_source_schedule_is_unevaluated_and_bilingual():
    cases=preregister_provisional_cases()
    assert len(cases)==OOD_COUNT
    assert len({x["index"] for x in cases})==OOD_COUNT
    assert all(x["split"]=="OOD_CANDIDATE_NOT_FROZEN" for x in cases)
    assert sum(x["language"]=="ar" for x in cases)==16
    assert sum(x["language"]=="en" for x in cases)==16

@pytest.mark.parametrize("missing",[
    "new_ood_gold_free_source_frozen",
    "independent_pixel_replays",
    "native_reference_successes",
    "before_model_inferences","after_model_inferences",
    "native_one_shot_rewards","old_eval_used",
    "dev_gold_on_gpu","paid_gpu"])
def test_smoke_only_is_never_an_ood_result(missing):
    evidence=dict(new_ood_gold_free_source_frozen=True,
      independent_pixel_replays=32,native_reference_successes=32,
      before_model_inferences=32,after_model_inferences=32,
      native_one_shot_rewards=64,old_eval_used=False,
      dev_gold_on_gpu=False,paid_gpu=False)
    del evidence[missing]
    with pytest.raises(ValueError):
        assert_ood_not_qualified_without_new_receipts(evidence)

def test_all_future_gate_fields_required_before_claiming_true_ood():
    evidence=dict(new_ood_gold_free_source_frozen=True,
      independent_pixel_replays=32,native_reference_successes=32,
      before_model_inferences=32,after_model_inferences=32,
      native_one_shot_rewards=64,old_eval_used=False,
      dev_gold_on_gpu=False,paid_gpu=False)
    assert assert_ood_not_qualified_without_new_receipts(evidence) is True
