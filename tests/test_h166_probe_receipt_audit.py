"""Fail-closed H166 GPU receipt integrity tests on synthetic fixture objects."""
import json
from pathlib import Path
import pytest
from h166.audit_probe import audit,SOURCE_COMMIT
from h132.grounder_contract import MODEL_ID,MODEL_SHA

def fixture(tmp_path):
    output=tmp_path/"hala-h166-synthetic-vision-probe"
    output.mkdir()
    job={"schema":"hala.h166.private-synthetic-visual-t4-job.v1",
         "status":"H166_SYNTHETIC_FREE_T4_MEMORY_STEP_PASS",
         "source_commit":SOURCE_COMMIT,"optimizer_steps":1,
         "real_app_training":False,"owner_gold_used":False,
         "new_ood_evaluations":0,"model_weights_published":False,
         "paid_GPU":False,"h140_blind_calls":0,"peak_allocated_gib":8.25,
         "trainable_parameters":512}
    model={"schema":"hala.h166.synthetic-visual-merger-t4-probe.v1",
         "status":"H166_ONE_SYNTHETIC_VISUAL_MERGER_STEP_PASSED",
         "model_id":MODEL_ID,"model_revision":MODEL_SHA,"gpu":"Tesla T4",
         "environment":{"torch":"2.11.0","transformers":"5.17.0",
                        "peft":"0.21.2","Pillow":"11.3.0"},
         "optimizer_steps":1,"synthetic_only":True,
         "source_original_app_images":0,"owner_gold_used":False,
         "new_ood_evaluations":0,"model_weights_saved":False,
         "paid_GPU":False,"h140_blind_calls":0,
         "flagship_35B_unchanged":True,"lora_rank":4,
         "memory_cap_gib":13.0,
         "visual_merger_modules":["model.visual.merger.mlp.0"],
         "trainable_parameters":512,"nonzero_gradient_tensors":1,
         "updated_parameter_tensors":1,"optimizer_states":2,
         "finite_synthetic_loss":True,"peak_allocated_gib":8.25,
         "peak_reserved_gib":8.5}
    (tmp_path/"h166-job-receipt.json").write_text(json.dumps(job))
    (output/"h166-visual-memory-receipt.json").write_text(json.dumps(model))
    return job,model

def save(tmp_path,job,model):
    (tmp_path/"h166-job-receipt.json").write_text(json.dumps(job))
    (tmp_path/"hala-h166-synthetic-vision-probe"/"h166-visual-memory-receipt.json").write_text(json.dumps(model))

def test_baseline_fixture_is_syntactically_qualified(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)["optimizer_steps"]==1

@pytest.mark.parametrize("field,value",[
    ("synthetic_only",False),
    ("source_original_app_images",1),
    ("model_weights_saved",True),
    ("paid_GPU",True),
    ("new_ood_evaluations",1),
    ("visual_merger_modules",["model.language_model.layers.0.mlp"]),
    ("peak_allocated_gib",13.01),
    ("updated_parameter_tensors",0),
    ("nonzero_gradient_tensors",0)])
def test_model_receipt_mutation_fails(tmp_path,field,value):
    job,model=fixture(tmp_path)
    model[field]=value
    save(tmp_path,job,model)
    with pytest.raises(ValueError):
        audit(tmp_path)

@pytest.mark.parametrize("field,value",[
    ("source_commit","bad"),("paid_GPU",True),
    ("optimizer_steps",2),("new_ood_evaluations",1)])
def test_job_receipt_mutation_fails(tmp_path,field,value):
    job,model=fixture(tmp_path)
    job[field]=value
    save(tmp_path,job,model)
    with pytest.raises(ValueError):
        audit(tmp_path)
