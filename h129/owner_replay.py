"""Owner-independent H129 private SHA/neural/paired-source-score replay."""
import hashlib
import json
from pathlib import Path
from h125.model_contracts import MODELS,parse_point
from h129.feedback import score

ARM="showui_2b"

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def replay(source,raw,reported,raw_sha):
    if (source.get("schema")!="hala.h129.owner-private-true-original-TRAIN16-source-replay.v1" or
        source.get("source_schedule_sha256")!=reported.get("source_schedule_sha256") or
        source.get("source_cases_sha256")!=reported.get("source_cases_sha256") or
        source.get("original_image_byte_sha_verified")!=16 or
        source.get("original_previous_excluded")!=2704 or
        reported.get("model_id")!=MODELS[ARM].model_id or
        reported.get("model_revision")!=MODELS[ARM].revision_sha or
        reported.get("real_neural_calls")!=64 or
        reported.get("private_raw_sha256")!=raw_sha or
        reported.get("is_real_interactive_environment") is not False):
        raise ValueError("H129 private source/model/native-evidence provenance mismatch")
    cases=source.get("cases")
    if not isinstance(cases,list) or len(cases)!=16 or not isinstance(raw,list) or len(raw)!=16:
        raise ValueError("H129 exact private paired 16 real original scene responses required")
    rec=[]
    calls=0
    for c,item in zip(cases,raw):
        if c["id"]!=item.get("id") or c["image_sha256"]!=item.get("image_sha256"):
            raise ValueError("H129 source / neural media SHA or ID drift")
        w=c["image_width"];h=c["image_height"]
        result={"id":c["id"],"image_sha256":c["image_sha256"]}
        for role in ("start","end"):
            direct=parse_point(ARM,item.get("direct_raw_"+role),w,h)
            feedback=parse_point(ARM,item.get("feedback_raw_"+role),w,h)
            if direct is None or feedback is None:
                raise ValueError("H129 native ShowUI syntax invalid")
            try:
                output_direct=tuple(item["direct_"+role])
                output_feedback=tuple(item["feedback_"+role])
            except (KeyError,TypeError):
                raise ValueError("H129 absent original neural output coordinates")
            if direct!=output_direct or feedback!=output_feedback:
                raise ValueError("H129 native real neural parsed coordinate mismatch")
            result["direct_"+role]=direct
            result["feedback_"+role]=feedback
            calls+=2
        rec.append(result)
    if calls!=64:
        raise ValueError("H129 64 actual neural responses required")
    measured=score(cases,rec)
    if measured["aggregate"]!=reported.get("aggregate") or measured["per_family"]!=reported.get("per_family"):
        raise ValueError("H129 GPU scored metrics not independently reproducible")
    return {"schema":"hala.h129.owner-independent-original-native-feedback-replay.v1",
            "status":"PASS_H129_NATIVE_SOURCE_SHA_PAIRED_SCORE_REPLAY",
            "source_schedule_sha256":source["source_schedule_sha256"],
            "model_id":reported["model_id"],
            "model_revision":reported["model_revision"],
            "raw_neural_sha256":raw_sha,
            "verified_cases":16,"verified_model_calls":64,
            "independent_recomputed_metrics":measured["aggregate"],
            "model_skill_qualified":False,
            "original_screenshots_or_weights_published":False}

def run_files(source_file,raw_file,metrics_file,destination):
    destination=Path(destination)
    if destination.exists():
        raise ValueError("H129 owner independent replay create-only")
    source=json.loads(Path(source_file).read_text("utf8"))
    raw=json.loads(Path(raw_file).read_text("utf8"))
    metrics=json.loads(Path(metrics_file).read_text("utf8"))
    result=replay(source,raw,metrics,file_hash(raw_file))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
