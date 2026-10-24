"""H117 weighted next-token target only for spatial signed delta or start point.

The older coordinate-digit weighting incorrectly also weights digits of
the deterministic coordinate_space schema string normalized_0_1000.
For H117 only, mask numeric values for point or signed delta, not metadata.
"""
from __future__ import annotations
import re

def target_only_weights(token_ids,decode,role,coordinate_weight=8.0):
    if role not in ("start","end_delta"):
        raise ValueError("H117 one of the two signed-role target objectives required")
    if (type(coordinate_weight) not in (int,float)
            or not 1.0<=coordinate_weight<=64.0
            or not token_ids or any(type(t) is not int for t in token_ids)):
        raise ValueError("H117 safe supervised coordinate tokens required")
    pieces=[decode([t]) for t in token_ids]
    if any(not isinstance(x,str) for x in pieces):
        raise ValueError("H117 decoder must return text")
    content="".join(pieces)
    # Accept only exactly the appropriate coordinate field; no metadata digits.
    name="point" if role=="start" else "delta"
    sign="" if role=="start" else "-?"
    match=re.findall(r'"'+name+r'"\s*:\s*\[\s*('+sign+r'\d+)\s*,\s*('+sign+r'\d+)\s*\]',content)
    if len(match)!=1:
        raise ValueError("H117 target coordinate array not exactly one")
    pattern=re.compile(r'"'+name+r'"\s*:\s*\[\s*('+sign+r'\d+)\s*,\s*('+sign+r'\d+)\s*\]')
    m=pattern.search(content)
    ranges=[m.span(1),m.span(2)]
    result=[]
    offset=0
    for part in pieces:
        end=offset+len(part)
        hit=any(offset<right and end>left for left,right in ranges)
        result.append(float(coordinate_weight) if hit else 1.0)
        offset=end
    if not any(x>1 for x in result):
        raise ValueError("H117 no normalized point or signed delta tokens supervised")
    return result
