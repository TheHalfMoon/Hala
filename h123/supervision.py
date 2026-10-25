"""H123 native absolute-END spatial grid token supervision.

Weight ONLY the three original source-derived numeric target arrays,
never task text, metadata, punctuation or source-derived label IDs.
"""
import re

FIELDS=("quadrant","cell","point")


def spatial_weights(token_ids,decode,coordinate_weight=8.0):
    if (type(token_ids) is not list or not token_ids
            or any(type(i) is not int for i in token_ids)
            or type(coordinate_weight) not in (int,float)
            or not 1<=coordinate_weight<=64):
        raise ValueError("H123 bounded exact 3-stage END target required")
    parts=[decode([x]) for x in token_ids]
    if any(type(part) is not str for part in parts):
        raise ValueError("H123 token decoding must be deterministic text")
    full="".join(parts)
    spans=[]
    for field in FIELDS:
        m=list(re.finditer('"'+field+r'"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',full))
        if len(m)!=1:
            raise ValueError("H123 exact one quadrant/cell/point numeric target needed")
        spans.extend((m[0].span(1),m[0].span(2)))
    weights=[];offset=0
    for part in parts:
        stop=offset+len(part)
        weights.append(float(coordinate_weight) if any(
            offset<end and stop>begin for begin,end in spans) else 1.0)
        offset=stop
    if len(weights)!=len(token_ids) or not any(x>1 for x in weights):
        raise ValueError("H123 target-only weighted spatial digits missing")
    return weights
