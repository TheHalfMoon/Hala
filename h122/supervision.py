"""H122 two target fields coordinate/cell digits; metadata must stay weight=1."""
from __future__ import annotations
import re

def coarse_fine_weights(token_ids,decode,coordinate_weight=8.0):
    if (not isinstance(token_ids,list) or not token_ids
            or any(type(i) is not int for i in token_ids)
            or type(coordinate_weight) not in (float,int)
            or not 1.0<=coordinate_weight<=64.0):
        raise ValueError("H122 bounded full END role token supervision required")
    pieces=[decode([x]) for x in token_ids]
    if any(type(piece) is not str for piece in pieces):
        raise ValueError("H122 tokenizer must produce exact text")
    source="".join(pieces)
    matches=[]
    for key in ("cell","point"):
        found=list(re.finditer(r'"'+key+r'"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',source))
        if len(found)!=1:
            raise ValueError("H122 only exact one cell and point target array")
        matches.extend((found[0].span(1),found[0].span(2)))
    weight=[]
    offset=0
    for part in pieces:
        stop=offset+len(part)
        weight.append(float(coordinate_weight) if any(
            offset<end and stop>begin for begin,end in matches) else 1.0)
        offset=stop
    if not any(x>1 for x in weight):
        raise ValueError("H122 no spatial target coordinates supervised")
    return weight
