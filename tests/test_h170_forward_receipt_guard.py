"""H170 independent frozen visual effect report must not invent nonzero deltas."""
import pytest
from h170.audit_forward import qualify_delta

def test_finite_zero_delta_is_valid_null_result():
    assert qualify_delta(dict(max_abs_delta=0.0,l2_delta=0.0,
                              nonzero_elements=0,element_count=1024)) is False

def test_positive_logit_delta_is_numeric_effect_not_accuracy():
    assert qualify_delta(dict(max_abs_delta=0.003,l2_delta=0.015,
                              nonzero_elements=8,element_count=256)) is True

@pytest.mark.parametrize("change",[
    dict(max_abs_delta=1.0,l2_delta=0.0,nonzero_elements=0,element_count=2),
    dict(max_abs_delta=0.0,l2_delta=0.0,nonzero_elements=1,element_count=2),
    dict(max_abs_delta=1.0,l2_delta=1.0,nonzero_elements=3,element_count=2),
    dict(max_abs_delta=-1.0,l2_delta=1.0,nonzero_elements=1,element_count=2),
    dict(max_abs_delta=1.0,l2_delta=1.0,nonzero_elements=1,element_count=0),
    dict(max_abs_delta=float("nan"),l2_delta=1.0,nonzero_elements=1,element_count=2)])
def test_mutated_or_nonfinite_forward_delta_rejected(change):
    with pytest.raises(ValueError):
        qualify_delta(change)
