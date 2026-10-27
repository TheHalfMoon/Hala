"""H164 typed computer-use action contract: native click, drag, text, abstain.

No DOM selectors, XPath, browser JS, script execution or private oracle data
may appear in model outputs. This is an offline schema, not a trained model.
"""
import json
import math

KINDS=("click","drag","type","abstain")
WIDTH=960
HEIGHT=720
MAX_TEXT=128

def _point(value):
    if (type(value) is not list or len(value)!=2 or
        any(type(x) not in (int,float) or not math.isfinite(x)
            or not 0<=x<bound
            for x,bound in zip(value,(WIDTH,HEIGHT)))):
        raise ValueError("H164 absolute original screenshot pixel outside viewport")
    return value

def parse_action(raw):
    if type(raw) is not str or len(raw)>1400:
        return {"valid":False,"kind":None}
    try:
        data=json.loads(raw)
        if type(data) is not dict or set(data)!={"actions"}:
            raise ValueError("H164 one typed JSON action only")
        arr=data["actions"]
        if type(arr) is not list or len(arr)!=1 or type(arr[0]) is not dict:
            raise ValueError("H164 exactly one native action required")
        action=arr[0]
        kind=action.get("kind")
        if kind=="abstain":
            if set(action)!={"kind"}:
                raise ValueError("H164 abstain may not include other fields")
            return {"valid":True,"kind":"abstain","points":[]}
        if kind=="click":
            if set(action)!={"kind","point"}:
                raise ValueError("H164 click may only specify screenshot point")
            points=[_point(action["point"])]
        elif kind=="drag":
            if set(action)!={"kind","start","end"}:
                raise ValueError("H164 drag only includes start/end screenshot points")
            points=[_point(action["start"]),_point(action["end"])]
        elif kind=="type":
            if set(action)!={"kind","point","text"}:
                raise ValueError("H164 native type has point+text only")
            points=[_point(action["point"])]
            text=action["text"]
            if (type(text) is not str or not text or len(text)>MAX_TEXT or
                any(ord(char)<32 for char in text)):
                raise ValueError("H164 native typing payload invalid")
        else:
            raise ValueError("H164 unsupported native action kind")
        result={"valid":True,"kind":kind,"points":points}
        if kind=="type":
            result["text"]=action["text"]
        return result
    except (ValueError,TypeError,KeyError,OverflowError):
        return {"valid":False,"kind":None}

def supervision(row):
    doc=row.get("training_supervision")
    if type(doc) is not dict:
        raise ValueError("H164 gold supervision restricted to TRAIN only")
    encoded=json.dumps(doc,sort_keys=True,ensure_ascii=False,
                       separators=(",",":"))
    proposal=parse_action(encoded)
    if (proposal["valid"] is not True or
        proposal["kind"]!=row.get("action_type") or
        proposal["kind"]=="abstain"):
        raise ValueError("H164 real UI TRAIN action doesn't match typed schema")
    return encoded

def visible_prompt(row):
    if "training_supervision" not in row and row.get("split")=="TRAIN":
        raise ValueError("H164 missing TRAIN gold to pair task/prompt")
    task=row.get("instruction")
    if type(task) is not str or not task.strip() or len(task)>500:
        raise ValueError("H164 original app task absent or excessive")
    return (
        "You are Hala, a visual computer-use agent. Read the original "
        "960x720 screenshot and the user's task. Output ONLY one JSON "
        "object with a single actions array: "
        '{"actions":[{"kind":"click","point":[x,y]}]} or '
        '{"actions":[{"kind":"drag","start":[x,y],"end":[x,y]}]} or '
        '{"actions":[{"kind":"type","point":[x,y],"text":"..."}]} or '
        '{"actions":[{"kind":"abstain"}]}. '
        "All points are absolute original screenshot pixels. "
        "Do not guess HTML selectors, hidden page code or owner gold. "
        "If unsure, abstain. Task: " + task
    )
