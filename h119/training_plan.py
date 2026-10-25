"""H119 exact full H105 512 original TRAIN screenshots, source-bound."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
from h61.curriculum import read_shard, shard_roots
from h105.curriculum import propose, source_geometry, aggregate
from h106.training_plan import H105_QUALIFIED_PLAN_SHA

STEPS=1024
PAIRS=512

@dataclass(frozen=True)
class FullPlan:
    indices: tuple[int,...]
    sha256: str
    summary: dict

def get_plan(rows,training_root:Path,steps:int=STEPS):
    if type(steps) is not int or steps!=STEPS or len(rows)!=10000:
        raise ValueError("H119 exact 1024 TRAIN-only start/end optimizer steps")
    sources={}
    root=Path(training_root)
    for shard in shard_roots(root):
        prefix=shard.relative_to(root).as_posix() if shard!=root else ""
        originals,_=read_shard(shard,prefix)
        for row in originals:
            if row["id"] in sources:
                raise ValueError("duplicate original H71 TRAIN source ID")
            sources[row["id"]]=row
    if len(sources)<10000:
        raise ValueError("original H71 TRAIN source incomplete")
    proposal=propose(rows,sources,PAIRS)
    if proposal.sha256!=H105_QUALIFIED_PLAN_SHA or not proposal.summary["all_gates_passed"]:
        raise ValueError("exact old H105 512 selected TRAIN source SHA differs")
    selected=[source_geometry(rows[i],sources[rows[i]["id"]]) for i in proposal.indices]
    if (len(selected)!=PAIRS or any(x is None for x in selected)
            or len({x["image_sha256"] for x in selected})!=PAIRS):
        raise ValueError("H119 original image source uniqueness mismatch")
    report=aggregate(selected)
    if (report["unique_images"]!=PAIRS or report["language"].get("ar",0)<100
            or len(report["family"])!=5 or report["largest_direction_fraction"]>=.4):
        raise ValueError("H119 full512 original bilingual family spatial TRAIN diversity gate")
    digest=hashlib.sha256(("hala.h119.full512:"+proposal.sha256).encode()).hexdigest()
    report.update(source_split="TRAIN_ONLY",parent_original_h105_sha256=proposal.sha256,
                  actual_source_screenshots=PAIRS,planned_optimizer_steps=STEPS,
                  no_eval_or_test=True,raw_source_ids_exposed=False,plan_sha256=digest)
    return FullPlan(tuple(proposal.indices),digest,report)


def checkpoint_due(step:int,max_steps:int=STEPS)->bool:
    """Cap H119 private output at sixteen durable optimizer snapshots."""
    if (type(step) is not int or type(max_steps) is not int or
            max_steps!=STEPS or not 1<=step<=max_steps):
        raise ValueError("H119 only exact 1024-step training may thin snapshots")
    return step%64==0 or step==max_steps
