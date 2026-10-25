"""H124 exact source joint START+END+signed delta numeric loss emphasis."""
import re
FIELDS=("start_point","end_point","start_cell","end_cell","delta")

def spatial_weights(token_ids,decode,coordinate_weight=8.0):
    if (type(token_ids) is not list or not token_ids
            or any(type(x) is not int for x in token_ids)
            or type(coordinate_weight) not in (int,float)
            or not 1<=coordinate_weight<=64):
        raise ValueError("H124 exact bounded coordinate tokens required")
    decoded=[decode([i]) for i in token_ids]
    if any(type(s) is not str for s in decoded):
        raise ValueError("H124 deterministic model tokenizer decoding required")
    text="".join(decoded)
    spans=[]
    for name in FIELDS:
        regex='"'+name+r'"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        matches=list(re.finditer(regex,text))
        if len(matches)!=1:
            raise ValueError("H124 each START END cell and signed delta numeric pair required exactly once")
        spans.extend([matches[0].span(1),matches[0].span(2)])
    weights=[];pos=0
    for part in decoded:
        nxt=pos+len(part)
        weights.append(float(coordinate_weight) if any(
            pos<end and nxt>begin for begin,end in spans) else 1.0)
        pos=nxt
    if len(weights)!=len(token_ids) or not any(x>1.0 for x in weights):
        raise ValueError("H124 numeric target tokens missing")
    return weights
