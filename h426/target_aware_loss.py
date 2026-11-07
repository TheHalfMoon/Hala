"""H426 ORIGINAL native-drag coordinate token loss, proposed research ONLY.

Independent owner H421 source-error audit: end-point-only failures in
slide_resize (63/128 actions), both-point failures in text_highlight (60/128).
This pure helper computes field-specific numeric grounding token weights
without training, GPU inference, or changing H425. An externally sealed
bilingual evaluation and independent screenshot rights are NOT satisfied.
"""
from __future__ import annotations

import json
import re

BASE_WEIGHT=1.0
COORD_WEIGHT=8.0
START_HIGHLIGHT_WEIGHT=12.0
END_RESIZE_WEIGHT=16.0
DOMAINS=frozenset(("slider","slide_resize","text_highlight","sheet"))


def source_coordinate_token_weights(pieces: list[str], domain: str) -> list[float]:
    if domain not in DOMAINS:
        raise ValueError("H426 unknown original source domain")
    if not pieces or any(type(x) is not str for x in pieces):
        raise ValueError("H426 supervised token pieces missing")
    text="".join(pieces)
    start=text.find('{"type":"drag"')
    end=text.find("}",start)+1
    if start<0 or end<=start or text.find('{"type":"drag"',start+1)>=0:
        raise ValueError("H426 expected exactly one native drag supervised JSON")
    raw=text[start:end]
    # The H425 assistant target is exactly ONE native-drag JSON object, optionally
    # followed by Qwen's end-of-message token. Extra prefixes/suffixes must not
    # be silently optimized as supervised answer content.
    if start!=0 or text[end:] not in ("", "<|im_end|>", "<|im_end|>\n"):
        raise ValueError("H426 supervised target contains unrecognized prefix or suffix")
    def reject_duplicate_keys(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:
                raise ValueError("H426 duplicate supervised native JSON field")
            obj[key]=value
        return obj
    try:
        # Normal json.loads silently accepts duplicate coordinate fields and
        # uses the LAST occurrence, while regex below selects the FIRST.
        # Reject ambiguous labels instead of assigning weight to wrong gold.
        data=json.loads(raw,object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError,TypeError) as e:
        raise ValueError("H426 original supervised JSON action not parseable") from e
    if (type(data) is not dict or set(data)!={
            "type","start_point","end_point","ordered","coordinate_space"}
        or data["type"]!="drag" or data["coordinate_space"]!="normalized_1000"
        or type(data["ordered"]) is not bool):
        raise ValueError("H426 supervised native drag JSON grammar drift")
    for key in ("start_point","end_point"):
        value=data[key]
        if (type(value) is not list or len(value)!=2
            or any(type(x) is not int or x<0 or x>1000 for x in value)):
            raise ValueError("H426 original native source coordinate outside 0..1000")
    numeric={}
    for field in ("start_point","end_point"):
        match=re.search(r'"'+field+r'":\[(\d{1,4}),(\d{1,4})\]',raw)
        if match is None:
            raise ValueError("H426 supervised source coordinate token boundary drift")
        spans=[]
        for k in (1,2):
            spans.append((start+match.start(k),start+match.end(k)))
        numeric[field]=spans
    ranges=[]
    cursor=0
    for segment in pieces:
        ranges.append((cursor,cursor+len(segment)))
        cursor+=len(segment)
    result=[]
    start_hits=0
    end_hits=0
    for begin,end_pos in ranges:
        detected=set()
        for field,spans in numeric.items():
            if any(max(a,begin)<min(b,end_pos) for a,b in spans):
                detected.add(field)
        if len(detected)>1:
            raise ValueError("H426 token spans BOTH coordinate fields; cannot safely weight")
        if not detected:
            result.append(BASE_WEIGHT)
            continue
        # A merged BPE token may contain coordinate AND noncoordinate digits.
        # Such a token cannot receive an unambiguous source-coordinate weight.
        # Fail closed instead of overweighting numeric template metadata.
        allowed=[(a,b) for f in detected for a,b in numeric[f]]
        if any(text[j].isdigit() and not any(a<=j<b for a,b in allowed)
               for j in range(begin,end_pos)):
            raise ValueError("H426 token merges target coordinates with noncoordinate numerals")
        field=next(iter(detected))
        if field=="end_point":
            end_hits+=1
            result.append(END_RESIZE_WEIGHT if domain=="slide_resize" else
                          START_HIGHLIGHT_WEIGHT if domain=="text_highlight" else COORD_WEIGHT)
        else:
            start_hits+=1
            result.append(START_HIGHLIGHT_WEIGHT if domain=="text_highlight" else COORD_WEIGHT)
    if start_hits<1 or end_hits<1:
        raise ValueError("H426 supervised native coordinate numeric tokens missing")
    if len(result)!=len(pieces) or not all(x>0 for x in result):
        raise ValueError("H426 calculated coordinate token weights invalid")
    return result


def proposed_h426_weighted_ce(output, labels, processor, domain: str):
    """Unexecuted H426 research experiment loss; no H425 training change."""
    import torch
    logits=output.logits[:,:-1,:]
    targets=labels[:,1:]
    keep=targets!=-100
    if not bool(keep.any()) or targets.size(0)!=1:
        raise ValueError("H426 gradient examples must be one source image+target")
    ids=targets[keep]
    scores=logits[keep].float()
    pieces=[processor.tokenizer.decode([int(i)],skip_special_tokens=False)
            for i in ids.tolist()]
    weights=source_coordinate_token_weights(pieces,domain)
    w=torch.tensor(weights,device=scores.device,dtype=torch.float32)
    ce=torch.nn.functional.cross_entropy(scores,ids,reduction="none")
    return (ce*w).sum()/w.sum()
