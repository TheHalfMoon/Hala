"""H172 new DEV preregistration is disjoint from old TRAIN/DEV variants."""
import pytest
from h172.source_dev import cases,NODE_PREFIXES,SCHEMA
from h171.source import cases as old_train

def test_fixed_new_dev32_source_and_bilingual_quota():
    rows=cases()
    assert len(rows)==32
    assert len({x["index"] for x in rows})==32
    assert all(x["split"]=="DEV" and x["action_type"]=="click"
               for x in rows)
    assert sum(x["language"]=="ar" for x in rows)==16
    assert sum(x["language"]=="en" for x in rows)==16
    assert {x["app"] for x in rows}=={"flatpickr","awesomplete"}
    assert set(x["index"] for x in rows).isdisjoint(
           set(x["index"] for x in old_train()))
    assert SCHEMA=="hala.h172.new-four-app-candidate-dev32-owner-only.v1"

def test_16_distinct_original_node_autocomplete_prefixes():
    assert len(NODE_PREFIXES)==16
    assert len(set(NODE_PREFIXES))==16
    assert all("node.js".startswith(p.lower()) for p in NODE_PREFIXES)
    assert "JavaScript" not in NODE_PREFIXES
