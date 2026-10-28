"""H168 fail-closed historical holdout and private native-training source tests."""
import copy
import pytest

from h168.source import SCHEMA,PINS,cases,validate_pilot

def fresh():
    r=cases()[:2]
    manifest={"schema":SCHEMA,"split":"TRAIN_PILOT","count":2,
              "private_cases":[dict(x,image_sha256=("%064x"%(n+16)),
                   native_browser_state_verified=True)
                   for n,x in enumerate(r)]}
    sup={"private_training_actions":[dict(index=x["index"],kind="type",
                 text="new_tag",point=[123,456],native_success=True)
                  for x in r]}
    return manifest,sup

def test_scheduled_cases_are_bilingual_two_new_app_cohorts():
    c=cases()
    assert len(c)==32
    assert len({x["index"] for x in c})==32
    assert {x["app"] for x in c}=={"choices","nouislider"}
    assert len([x for x in c if x["language"]=="ar"])==16
    assert len([x for x in c if x["action_type"]=="type"])==16
    assert len([x for x in c if x["action_type"]=="drag"])==16
    assert all(x["split"]=="TRAIN_PILOT" for x in c)
    assert len({spec[0] for spec in PINS.values()})==3

def test_private_source_provenance_admission_accepted():
    m,s=fresh()
    assert validate_pilot(m,s,set())["cases"]==2

@pytest.mark.parametrize("mutator",[
    lambda m,s: m.update(split="OOD"),
    lambda m,s: m["private_cases"][0].update(image_sha256="old"),
    lambda m,s: m["private_cases"][0].update(image_sha256=m["private_cases"][1]["image_sha256"]),
    lambda m,s: m["private_cases"][0].update(training_supervision={"actions":[]}),
    lambda m,s: m["private_cases"][0].update(native_browser_state_verified=False),
    lambda m,s: m["private_cases"][0].update(action_type="click"),
    lambda m,s: s["private_training_actions"][0].update(native_success=False),
    lambda m,s: s["private_training_actions"][0].update(index=-1),
    lambda m,s: s["private_training_actions"][0].update(kind="drag"),
    lambda m,s: s["private_training_actions"][0].update(point=[1]),
    lambda m,s: m["private_cases"].append(copy.deepcopy(m["private_cases"][0])),
])
def test_mutated_source_or_owner_action_fails_closed(mutator):
    m,s=fresh()
    mutator(m,s)
    with pytest.raises(ValueError):
        validate_pilot(m,s,set())

def test_historical_consumed_image_refused_by_sha_only():
    m,s=fresh()
    old=m["private_cases"][0]["image_sha256"]
    with pytest.raises(ValueError):
        validate_pilot(m,s,{old})
