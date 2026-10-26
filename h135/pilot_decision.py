"""Predeclared H135 frozen 2-resolution pilot selection.

Primary criterion full GUI tasks with adapter, secondary START handle hits,
third median START pixel error; exact ties favor native model-card behavior.
This is experimental mode selection, never a statistical release gate.
"""
import json
from pathlib import Path

def choose(outcome):
    if (outcome.get("status")!="H135_PILOT_PRIVATE_96_NEURAL_AND_REAL_BROWSER_SCORED" or
        outcome.get("real_neural_calls")!=96 or
        outcome.get("untouched_H135_sealed_holdout") is not True):
        raise ValueError("H135 must choose once from qualified independent real browser pilot")
    models=outcome.get("models",{})
    candidates={}
    for mode in ("downsample_640","native_960"):
        m=models.get("h134_adapter__"+mode)
        if not isinstance(m,dict) or m.get("n")!=12:
            raise ValueError("H135 both 12-scene adapter arms required")
        successes=m.get("real_gui_successes")
        hits=m.get("start_hits")
        median=m.get("median_start_error_px")
        if (type(successes) is not int or type(hits) is not int or
            not 0<=successes<=12 or not 0<=hits<=12 or
            type(median) not in (int,float) or median<0):
            raise ValueError("H135 invalid pilot score/median")
        candidates[mode]=(successes,hits,-median,mode=="native_960")
    chosen=max(candidates,key=lambda mode:candidates[mode])
    return {
        "schema":"hala.h135.once-only-selected-native-resolution.v1",
        "status":"H135_PILOT_FROZEN_MODEL_CARD_PREPROCESSING_MODE",
        "selected_mode":chosen,
        "decision_rule":"max(successes,START_hits,-median_START_error,native_tiebreak)",
        "native_960_rank_tuple":candidates["native_960"],
        "downsample_640_rank_tuple":candidates["downsample_640"],
        "pilot_scenes_used":12,
        "sealed_scenes_seen":0,
        "further_training_steps":0,
        "generalized_model_improvement_proven":False,
    }

def freeze(pilot_file,outfile):
    data=json.loads(Path(pilot_file).read_text("utf8"))
    decision=choose(data)
    outfile=Path(outfile)
    if outfile.exists():raise ValueError("H135 selection is create-only, no post-hoc mode choice")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(decision,sort_keys=True,indent=2)+"\n","utf8")
    return decision
