"""H169 independent exact paired native-reward statistics do not overclaim."""
import pytest
from h169.audit_reward import exact_mcnemar
from h169.source_dev import cases

def test_exact_no_discordant_pairs_is_null():
    assert exact_mcnemar(0,0)==1.0

def test_exact_four_wins_zero_regression():
    assert exact_mcnemar(4,0)==0.125

def test_symmetric_paired_discordance():
    assert exact_mcnemar(3,1)==exact_mcnemar(1,3)
    assert 0<=exact_mcnemar(3,1)<=1

def test_source_is_new_dev_not_trained_or_old_ood():
    all_cases=cases()
    assert len(all_cases)==32
    assert all(x["split"]=="DEV" and 16<=x["variant"]<=31
               for x in all_cases)
    assert len({x["index"] for x in all_cases})==32
    assert {x["app"] for x in all_cases}=={"choices","nouislider"}
