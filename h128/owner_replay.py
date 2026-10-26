"""Independent owner-private H128 neural-native output and strict score replay.

Verifies source SHA identity, direct and crop raw native parsing, baseline
anchoring independent of reference boxes, and independently re-computes
the source-bound paired score. NEVER attach private raw source data to PR.
"""
import hashlib
import json
from pathlib import Path
from h125.model_contracts import parse_point,MODELS
from h128.zoom import anchored_crop,native_to_original,score

ARM="showui_2b"

def canonical_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for data in iter(lambda:file.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()

def replay(source,raw,reported,raw_sha):
    if (source.get("schema")!="hala.h128.owner-private-true-original-TRAIN16-source-replay.v1" or
        source.get("source_schedule_sha256")!=reported.get("schedule_sha256") or
        source.get("source_cases_sha256")!=reported.get("source_cases_sha256") or
        source.get("original_image_byte_sha_verified")!=16 or
        source.get("original_previous_excluded")!=2688 or
        reported.get("model_id")!=MODELS[ARM].model_id or
        reported.get("model_revision")!=MODELS[ARM].revision_sha or
        reported.get("real_neural_calls")!=64 or
        reported.get("private_raw_sha256")!=raw_sha):
        raise ValueError("H128 original source/model/raw SHA provenance mismatch")
    cases=source.get("cases")
    if not isinstance(cases,list) or len(cases)!=16 or not isinstance(raw,list) or len(raw)!=16:
        raise ValueError("H128 exact private original 16 paired responses required")
    predictions=[]
    calls=0
    for case,item in zip(cases,raw):
        if (case["id"]!=item.get("id") or
            case["image_sha256"]!=item.get("image_sha256")):
            raise ValueError("H128 model neural/source case mismatch")
        width,height=case["image_width"],case["image_height"]
        out={"id":case["id"],"image_sha256":case["image_sha256"]}
        for role in ("start","end"):
            direct=parse_point(ARM,item.get("direct_raw_"+role),width,height)
            if direct is None or tuple(item.get("direct_"+role,()))!=direct:
                raise ValueError("H128 original direct pretrained neural output not reproducible")
            crop=anchored_crop(width,height,direct,ratio=0.5)
            if item.get("crop_"+role)!=[crop.left,crop.top,crop.right,crop.bottom]:
                raise ValueError("H128 crop not derived solely from direct pretrained prediction")
            local=parse_point(ARM,item.get("zoom_raw_"+role),crop.width,crop.height)
            expected=native_to_original(crop,local,width,height) if local else None
            prior=item.get("zoom_"+role)
            if (prior is None)!=(expected is None) or (prior is not None and tuple(prior)!=expected):
                raise ValueError("H128 zoom-crop re-projection differs from native model response")
            out["direct_"+role]=direct
            out["crop_"+role]=crop
            out["zoom_"+role]=expected
            calls+=2
        predictions.append(out)
    if calls!=64:
        raise ValueError("H128 exactly 64 real neural response records needed")
    measured=score(cases,predictions)
    if measured.get("aggregate")!=reported.get("aggregate") or measured.get("per_family")!=reported.get("per_family"):
        raise ValueError("H128 independent paired source-bound scoring differs from GPU report")
    return {
      "schema":"hala.h128.owner-independent-real-neural-sha-replay.v1",
      "status":"PASS_H128_ORIGINAL_SHA_NATIVE_PAIR_AND_SCORE_REPLAY",
      "source_schedule_sha256":source["source_schedule_sha256"],
      "model_id":reported["model_id"],"model_revision":reported["model_revision"],
      "actual_raw_sha256":raw_sha,
      "cases":16,"real_model_calls_verified":64,
      "independent_metric_recompute":measured["aggregate"],
      "model_skill_qualified":False,
      "model_weights_or_private_screenshots_published":False
    }

def run_files(source_path,raw_path,report_path,output_path):
    output_path=Path(output_path)
    if output_path.exists():
        raise ValueError("H128 independent replay must be create-only")
    source=json.loads(Path(source_path).read_text("utf8"))
    raw=json.loads(Path(raw_path).read_text("utf8"))
    report=json.loads(Path(report_path).read_text("utf8"))
    result=replay(source,raw,report,canonical_sha(raw_path))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
