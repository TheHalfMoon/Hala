"""H43 deterministic simulated app state and independent postcondition evaluator.

A synthetic action is *not* a real GUI click. The final-state oracle never
reads the golden demonstration and refuses unrelated-state mutation.
"""
from __future__ import annotations

import copy
from h43.factory import validate

ACTIONS={"row_target":"press","row_decoy":"press","amount_field":"fill",
         "confirm_button":"press","unrelated_button":"press"}


def score_state(case:dict, state:dict, events:list[dict])->bool:
    """Final-state based verifier: not golden-action sequence equality."""
    if (type(state) is not dict or type(events) is not list or
        set(state)!={"selected","amount","confirmed","unrelated_changed"} or
        type(state["confirmed"]) is not bool or
        type(state["unrelated_changed"]) is not bool):
        return False
    return (state==case["desired_state"] and not state["unrelated_changed"] and
            not any(event["ref"]=="unrelated_button" for event in events))


def simulate(case:dict, actions:list[dict])->dict:
    if (type(actions) is not list or len(actions)>case["max_actions"]
        or not all(type(a) is dict and set(a)=={"kind","ref","value"} for a in actions)):
        raise ValueError("Bounded, exact action contracts required")
    state=copy.deepcopy(case["initial_state"])
    events=[]
    for action in actions:
        ref=action["ref"]
        kind=action["kind"]
        value=action["value"]
        if type(ref) is not str or ref not in ACTIONS or kind!=ACTIONS[ref]:
            raise ValueError("Unrecognized or wrong-type simulated action")
        if kind=="press" and value is not None:
            raise ValueError("Button takes no invented payload")
        if kind=="fill" and (type(value) is not int or not 0<=value<=100):
            raise ValueError("A simulated quantity must be bounded integer")
        applied=True
        if ref=="row_target":
            state["selected"]=case["target_item"]
        elif ref=="row_decoy":
            state["selected"]=case["target_item"]+"-WRONG"
        elif ref=="amount_field":
            state["amount"]=value
        elif ref=="confirm_button":
            if state["selected"]==case["target_item"] and state["amount"]==case["requested_amount"]:
                state["confirmed"]=True
            else:
                applied=False
        elif ref=="unrelated_button":
            state["unrelated_changed"]=True
        events.append({"step":len(events)+1,"ref":ref,"kind":kind,"applied":applied})
    return {"state":state,"events":events,"success":score_state(case,state,events)}


def near_misses(case:dict)->dict[str,list[dict]]:
    golden=case["golden_actions"]
    def a(kind,ref,value=None):
        return {"kind":kind,"ref":ref,"value":value}
    return {
        "wrong_item":[a("press","row_decoy"),golden[1],golden[2]],
        "wrong_quantity":[golden[0],a("fill","amount_field",case["requested_amount"]+1),golden[2]],
        "skip_confirmation":[golden[0],golden[1]],
        "premature_confirmation":[golden[2],golden[0],golden[1]],
        "unrelated_mutation":[*golden,a("press","unrelated_button")],
        "wrong_item_after_selection":[golden[0],a("press","row_decoy"),golden[1],golden[2]],
    }


def validate_factory(manifest:dict)->dict:
    validate(manifest)
    cases=manifest["cases"]
    totals={"authoring":0,"dev":0}
    details=[]
    for case in cases:
        split=case["split"]
        totals[split]+=1
        baseline=simulate(case,[])
        golden=simulate(case,case["golden_actions"])
        checks={key:simulate(case,actions) for key,actions in near_misses(case).items()}
        if (baseline["success"] or not golden["success"]
            or golden["state"]!=case["desired_state"] or
            any(v["success"] for v in checks.values())):
            raise ValueError("H43 synthetic task oracle fails initial/golden/near-miss gate")
        details.append({"id":case["id"],"family":case["family"],
                        "language":case["language"],"split":split,
                        "baseline_rejected":True,"golden_reaches_goal":True,
                        "near_misses_rejected":sorted(checks),
                        "native_actions":False})
    return {"schema":"hala.h43.state-verifier-report.v1",
            "source_case_count":len(cases),"public_authoring_count":totals["authoring"],
            "public_dev_count":totals["dev"],"negative_replays":len(cases)*7,
            "positive_replays":len(cases),"cases":details,
            "model_trained":False,"gui_images":False,
            "independent_holdout_qualified":False,"native_actions":False}
