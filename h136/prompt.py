"""H136 original multi-theme GUI role prompts, color-invariant.

Prior blue/green-specific H131 prompts are invalid on 4 H136 palettes.
No oracle box, pixel labels or DOM target positions are provided to model.
"""
from h132.grounder_contract import PROMPT_TEMPLATE

def role_prompt(instruction,role):
    if role not in ("start","end"):
        raise ValueError("H136 permitted START/END role required")
    if not isinstance(instruction,str) or not 8<=len(instruction)<=500:
        raise ValueError("H136 original task instruction required")
    if role=="start":
        query=("Find the draggable control needed to BEGIN the described "
               "interaction. It may be a small slider thumb, resize grip, "
               "table separator or draggable card. "
               "Point to the center of the control, not its destination. "
               "Task: "+instruction)
    else:
        query=("Find the destination mark, outlined drop zone or endpoint "
               "needed to FINISH the described drag. "
               "Point to its center, not the draggable handle. "
               "Task: "+instruction)
    return PROMPT_TEMPLATE.format(instruction=query)
