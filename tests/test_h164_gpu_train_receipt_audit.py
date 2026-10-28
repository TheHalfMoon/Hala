"""Fail-closed synthetic tests for owner-private H164 GPU/DEV training auditor."""
import json
import pytest

from h164.audit_gpu_train import checked_rows
from h164.native_action_contract import parse_action

def rows(role="before"):
    result=[]
    for i in range(96):
        app="calculator" if i<48 else "simple-table"
        proposal=json.dumps({"actions":[{"kind":"click","point":[100+i,200]}]})
        result.append({
            "role":role,"app":app,"task":"synthetic action",
            "language":"en" if i%2 else "ar",
            "image_sha":format(i+1,"064x"),
            "raw":proposal,"parsed":parse_action(proposal)})
    return result

def dump(tmp_path,items):
    file=tmp_path/"private-dev.json"
    file.write_text(json.dumps(items),encoding="utf8")
    return file

def test_complete_frozen_96_dev_json_passes(tmp_path):
    records,counts,keys=checked_rows(dump(tmp_path,rows()),"before")
    assert len(records)==96 and len(keys)==96
    assert counts["calculator/valid"]==48
    assert counts["simple-table/valid"]==48

def test_same_image_sha_with_different_task_is_rejected(tmp_path):
    cases=rows()
    cases[95]["image_sha"]=cases[0]["image_sha"]
    cases[95]["task"]="different task must not bypass image identity"
    with pytest.raises(ValueError,match="same DEV image"):
        checked_rows(dump(tmp_path,cases),"before")

def test_mutated_action_parse_is_rejected(tmp_path):
    cases=rows()
    cases[5]["parsed"]={"kind":"click","valid":False}
    with pytest.raises(ValueError,match="model action"):
        checked_rows(dump(tmp_path,cases),"before")

def test_missing_dev_case_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="exact 96"):
        checked_rows(dump(tmp_path,rows()[:-1]),"before")

def test_mismatched_role_is_rejected(tmp_path):
    cases=rows()
    cases[31]["role"]="after"
    with pytest.raises(ValueError,match="model action"):
        checked_rows(dump(tmp_path,cases),"before")
