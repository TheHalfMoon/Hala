"""H171 source provenance, Native click gold isolation, bilingual quotas."""
import copy,hashlib
import pytest
from h171.source import SCHEMA,cases,validate,PREFIXES

def fixture():
    rows=[]
    gold=[]
    for row in cases():
        r=dict(row,image="images/"+str(row["index"])+".png",
            image_sha256=hashlib.sha256(
                ("synthetic-test-only-"+str(row["index"])).encode()).hexdigest(),
            instruction="Click the native original target",
            reference_native_success=True)
        rows.append(r)
        gold.append(dict(index=row["index"],kind="click",
                         point=[300,350],native_success=True))
    doc=dict(schema=SCHEMA,split="TRAIN_PILOT",count=32,
             no_prior_dev_or_ood_images=True,private_cases=rows)
    refs=dict(owner_train_actions=gold)
    return doc,refs

def test_new_original_click_plan_is_two_families_bilingual():
    rows=cases()
    assert len(rows)==32
    assert {x["app"] for x in rows}=={"flatpickr","awesomplete"}
    assert len(PREFIXES)==16 and len(set(PREFIXES))==16
    assert sum(x["language"]=="ar" for x in rows)==16
    assert all(x["split"]=="TRAIN_PILOT" and x["action_type"]=="click"
               for x in rows)

def test_valid_source_owner_action_separation():
    doc,gold=fixture()
    report=validate(doc,gold,set())
    assert report["cases"]==32
    assert report["owner_gold_separate"] is True

@pytest.mark.parametrize("mutate",[
    lambda d,g:d.update(split="OOD"),
    lambda d,g:d["private_cases"][0].update(image_sha256="not-sha"),
    lambda d,g:d["private_cases"][0].update(reference_native_success=False),
    lambda d,g:d["private_cases"][0].update(training_supervision={"actions":[]}),
    lambda d,g:d["private_cases"][0].update(app="old-pell"),
    lambda d,g:d["private_cases"][0].update(language="fr"),
    lambda d,g:g["owner_train_actions"][0].update(kind="drag"),
    lambda d,g:g["owner_train_actions"][0].update(native_success=False),
    lambda d,g:g["owner_train_actions"][0].update(point=[-1,350]),
    lambda d,g:g["owner_train_actions"][0].update(point=[500,722]),
    lambda d,g:g["owner_train_actions"][0].update(index=-1),
    lambda d,g:d["private_cases"][1].update(image_sha256=d["private_cases"][0]["image_sha256"])
])
def test_leakage_or_invalid_original_action_fails(mutate):
    doc,gold=fixture()
    mutate(doc,gold)
    with pytest.raises(ValueError):
        validate(doc,gold,set())

def test_consumed_dev_sha_rejected_without_opening_its_pixels():
    doc,gold=fixture()
    old=doc["private_cases"][0]["image_sha256"]
    with pytest.raises(ValueError):
        validate(doc,gold,{old})
