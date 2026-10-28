"""H167 fail-closed independent dual-receipt audit mutation tests."""
import json
import pytest
from h167.audit_combined import audit,SOURCE_COMMIT
from h167.combined_probe import (LANGUAGE_ZIP_SHA,LANGUAGE_WEIGHTS_SHA,
                                LANGUAGE_CONFIG_SHA,LANGUAGE_README_SHA)
from h132.grounder_contract import MODEL_ID,MODEL_SHA

def fixture(tmp_path):
    out=tmp_path/"hala-h167-private-combined-t4"
    out.mkdir()
    job={"schema":"hala.h167.private-combined-language-visual-kaggle-job.v1",
         "status":"H167_SYNTHETIC_COMBINED_FREE_T4_PASS",
         "source_commit":SOURCE_COMMIT,"optimizer_steps":2,
         "owner_gold_used":False,"previous_ood_or_dev_used":False,
         "new_ood_evaluations":0,"paid_gpu":False,
         "model_weights_published":False,"flagship_35b_unchanged":True,
         "peak_allocated_gib":11.4,"trainable_visual_parameters":57344}
    model={"schema":"hala.h167.private-frozen-h164-plus-vision-synthetic-t4.v1",
         "status":"H167_TWO_LANGUAGE_SYNTHETIC_STEPS_PASS",
         "language_adapter_zip_sha256":LANGUAGE_ZIP_SHA,
         "language_adapter_weights_sha256":LANGUAGE_WEIGHTS_SHA,
         "language_adapter_config_sha256":LANGUAGE_CONFIG_SHA,
         "language_adapter_readme_sha256":LANGUAGE_README_SHA,
         "language_adapter_frozen_before_after_match":True,
         "base_model":MODEL_ID,"base_revision":MODEL_SHA,
         "environment":{"torch":"2.11.0","transformers":"5.17.0",
                        "peft":"0.21.2","Pillow":"11.3.0"},
         "gpu":"Tesla T4","active_adapters":["language","vision"],
         "active_language_layers":2,
         "active_vision_modules":[
           "base_model.model.model.visual.merger.linear_fc1",
           "base_model.model.model.visual.merger.linear_fc2"],
         "visual_target_modules":["model.visual.merger.linear_fc1",
                                  "model.visual.merger.linear_fc2"],
         "visual_rank":4,"trainable_visual_parameters":57344,
         "frozen_language_parameters":1048576,
         "bilingual_synthetic":[{"language":lang,
                 "nonzero_grad_tensors":2,"updated_tensors":2,
                 "synthetic_finite_loss":True} for lang in ("en","ar")],
         "optimizer_steps":2,"peak_allocated_gib":11.4,
         "peak_reserved_gib":11.7,"memory_cap_gib":13.0,
         "data":"synthetic-only-no-original-app",
         "original_app_training_cases":0,"owner_gold_used":False,
         "previous_ood_or_dev_used":False,"new_ood_evaluations":0,
         "real_app_performance_measured":False,"weights_saved":False,
         "paid_gpu":False,"flagship_35b_unchanged":True}
    save(tmp_path,job,model)
    return job,model

def save(tmp_path,job,model):
    (tmp_path/"h167-combined-job-receipt.json").write_text(json.dumps(job))
    (tmp_path/"hala-h167-private-combined-t4"/
     "h167-combined-synthetic-t4-receipt.json").write_text(json.dumps(model))

def test_h167_two_receipt_fixture_accepts_only_qualified_structure(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)["frozen_language_unchanged"] is True

@pytest.mark.parametrize("field,value",[
    ("language_adapter_frozen_before_after_match",False),
    ("active_adapters",["vision"]),
    ("active_adapters",["language"]),
    ("previous_ood_or_dev_used",True),
    ("original_app_training_cases",1),
    ("owner_gold_used",True),
    ("weights_saved",True),
    ("peak_allocated_gib",13.05),
    ("visual_rank",8),
    ("trainable_visual_parameters",0),
    ("bilingual_synthetic",[]),
    ("bilingual_synthetic",[{"language":"en","nonzero_grad_tensors":2,
                           "updated_tensors":2,"synthetic_finite_loss":True}]*2)])
def test_tampered_model_receipt_rejected(tmp_path,field,value):
    job,model=fixture(tmp_path)
    model[field]=value
    save(tmp_path,job,model)
    with pytest.raises(ValueError):
        audit(tmp_path)

@pytest.mark.parametrize("field,value",[
    ("source_commit","000"),
    ("status","RUNNING"),
    ("optimizer_steps",0),
    ("new_ood_evaluations",1),
    ("model_weights_published",True),
    ("paid_gpu",True),
    ("peak_allocated_gib",9.0)])
def test_tampered_job_receipt_rejected(tmp_path,field,value):
    job,model=fixture(tmp_path)
    job[field]=value
    save(tmp_path,job,model)
    with pytest.raises(ValueError):
        audit(tmp_path)
