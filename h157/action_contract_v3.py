"""H157 extended native image-to-action contract for independent GUI layouts.

Existing source H156 DEV is consumed; H157 uses new original task scenes.
"""
import json
from h156.action_contract_v2 import action_from_dict_v2

SEQUENCES={
    "click-test":("click",),
    "click-button":("click",),
    "enter-text":("click","type","submit"),
    "enter-text-dynamic":("click","type","submit"),
    "choose-list":("select","submit"),
}
MAX_ACTIONS=5


def parse_v3(raw,*,task,model_checkpoint_sha):
    if (not isinstance(model_checkpoint_sha,str) or
        len(model_checkpoint_sha)!=64 or
        any(c not in "0123456789abcdef" for c in model_checkpoint_sha)):
        raise ValueError("H157 requires SHA-pinned action research checkpoint")
    if task not in SEQUENCES:
        raise ValueError("H157 unregistered original UI task")
    if type(raw) is not str or len(raw)>1600:
        raise ValueError("H157 exact bounded native action output only")
    try:
        document=json.loads(raw)
    except (TypeError,ValueError) as exc:
        raise ValueError("Model must emit exact JSON") from exc
    if type(document) is not dict or set(document)!={"actions"}:
        raise ValueError("Model must emit only actions array")
    data=document["actions"]
    if type(data) is not list or not 1<=len(data)<=MAX_ACTIONS:
        raise ValueError("Action count not preregistered")
    actions=tuple(action_from_dict_v2(x) for x in data)
    if any(a.kind=="abstain" for a in actions):
        if len(actions)!=1:
            raise ValueError("Abstention must be standalone")
        return actions
    if tuple(a.kind for a in actions)!=SEQUENCES[task]:
        raise ValueError("Incorrect task action order")
    return actions
