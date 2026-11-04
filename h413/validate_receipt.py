"""Strict original private H413 no-training receipt replay and provenance check."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from h413.kaggle.h413_new_family_transfer_eval import (
    BASE,REV,H409_SHA,H412_SHA,N,compose,source_contract,score,aggregate,
)

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h413-owner-private-family-transfer-eval"
STATUS="COMPLETE_PRIVATE_AUTHOR_VISIBLE_NEW_FAMILY_EVAL_NO_RELEASE"
FAILURE="FAILED_PRIVATE_SOURCE_ONLY_NO_RELEASE"


def _no_dupes(items):
    out={}
    for k,v in items:
        if k in out:
            raise ValueError("H413 duplicate receipt property")
        out[k]=v
    return out


def verify(path=ROOT, *, require_pixel_replay=True):
    if type(require_pixel_replay) is not bool:
        raise ValueError("H413 source replay mode must be explicit")
    root=Path(path)
    original=(root/"receipt.json").read_bytes()
    digest=hashlib.sha256(original).hexdigest()
    if digest!=(root/"receipt.sha256").read_text(encoding="ascii").strip():
        raise ValueError("H413 receipt hash mismatch")
    state=json.loads(original,object_pairs_hook=_no_dupes,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite")))
    if (type(state) is not dict
        or state.get("schema")!="hala.h413.private-original-new-family-eval.v1"
        or state.get("base")!=BASE or state.get("base_revision")!=REV
        or state.get("h409_adapter_sha256")!=H409_SHA
        or state.get("h412_adapter_sha256")!=H412_SHA
        or state.get("owner_first_party_generated_only") is not True
        or state.get("original_gui_train_rights_approved") is not False
        or state.get("independent_external_ood") is not False
        or state.get("official_hala_model_trained") is not False
        or state.get("weights_public") is not False
        or state.get("training_performed") is not False
        or state.get("model_release_qualified") is not False):
        raise ValueError("H413 mismatched lineage or fabricated release rights")
    if state.get("status")==FAILURE:
        if type(state.get("failure_class")) is not str:
            raise ValueError("H413 malformed failure")
        return {"receipt_sha256":digest,"status":FAILURE,"qualified":False}
    if state.get("status")!=STATUS or state.get("case_n")!=N:
        raise ValueError("H413 not a completed source experiment")
    if (state.get("device_name")!="Tesla T4" or
        type(state.get("elapsed_seconds")) not in (int,float) or
        not 0<state["elapsed_seconds"]<36000):
        raise ValueError("H413 GPU evidence incomplete")
    # Exact RGB pixels and gold boxes are independently recreated from the
    # source by this owner-local Python/Pillow verifier, never from model claims.
    rows=[compose(i) for i in range(N)]
    # On the original owner Kaggle/Windows rendering runtime the precise
    # glyph rasterization is reproducible and strict RGB comparison is
    # required. Linux CI uses an OS-provided DejaVuSans file with different
    # pixel glyphs, even when Pillow's version equals the training runtime.
    # The explicit fallback below NEVER CLAIMS TO REPLAY RGB: it anchors
    # both models' source IDs, native actions, geometry and recorded image
    # SHA set to the byte-verified ORIGINAL owner-private Kaggle receipt.
    # Full strict image verification remains the default and fail-closed.
    source_contract(rows)  # reject altered family/regime/source geometry
    if require_pixel_replay:
        if state.get("image_split_sha256")!=source_contract(rows):
            raise ValueError("H413 source visual fingerprints mismatch")
        pinned_hashes=[r["image_sha256"] for r in rows]
    else:
        original=state.get("h409_details")
        if type(original) is not list or len(original)!=N:
            raise ValueError("H413 no original source hashes for geometry-only audit")
        pinned_hashes=[]
        for rec in original:
            if type(rec) is not dict:
                raise ValueError("H413 source record missing")
            value=rec.get("image_sha256")
            if (type(value) is not str or len(value)!=64
                or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("H413 non-hex original source image hash")
            pinned_hashes.append(value)
        if (len(set(pinned_hashes))!=N or
            hashlib.sha256("".join(sorted(pinned_hashes)).encode()).hexdigest()!=
            state.get("image_split_sha256")):
            raise ValueError("H413 recorded original source hash set modified")
    out={}
    for name in ("h409","h412"):
        details=state.get(name+"_details")
        if type(details) is not list or len(details)!=N:
            raise ValueError("H413 incomplete unpaired generated cases")
        for i,(row,record) in enumerate(zip(rows,details)):
            if (type(record) is not dict or set(record)!={
                "id","family","kind","language","image_sha256",
                "raw_prediction","action","correct","unsafe_click","invalid"}):
                raise ValueError("H413 changed per-case schema")
            for field in ("id","family","kind","language"):
                if type(record[field]) is not str or record[field]!=row[field]:
                    raise ValueError("H413 case provenance or ordering mismatch")
            if (type(record["image_sha256"]) is not str or
                record["image_sha256"]!=pinned_hashes[i]):
                raise ValueError("H413 recorded RGB hash or model pair source mismatch")
            if type(record["raw_prediction"]) is not str:
                raise ValueError("H413 missing exact native generated output")
            proof=score(row,record["raw_prediction"])
            for field,value in proof.items():
                if type(record[field]) is not type(value) or record[field]!=value:
                    raise ValueError("H413 false generated click/abstain report")
        summary=aggregate(details)
        if state.get(name+"_audit")!=summary:
            raise ValueError("H413 aggregate mismatch from actual generated outputs")
        out[name]=summary
    return {"receipt_sha256":digest,"status":STATUS,"qualified":False,
            "source":"author-visible first-party different families; NOT external OOD",
            "source_rgb_replayed":require_pixel_replay,
            "h409":out["h409"],"h412":out["h412"]}


if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))
