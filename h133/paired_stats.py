"""H133 exact paired 8-scene descriptive checks, no power exaggeration."""
import math

def exact_discordant_p(specialist_only,showui_only):
    if (type(specialist_only) is not int or type(showui_only) is not int or
        specialist_only<0 or showui_only<0):
        raise ValueError("H133 nonnegative exact discordant paired counts required")
    total=specialist_only+showui_only
    if total==0: return 1.0
    extreme=min(specialist_only,showui_only)
    return min(1.0,2*sum(math.comb(total,k) for k in range(extreme+1))/(2**total))

def compare(models,metric):
    if metric not in ("actual_gui_task_success","start_hits_handle"):
        raise ValueError("H133 comparison uses frozen model skill metrics only")
    a=models["specialist"]["per_case"]
    b=models["showui"]["per_case"]
    if len(a)!=8 or len(b)!=8:
        raise ValueError("H133 exact eight task pairs required")
    old={x["case"]:x[metric] for x in b}
    if len(old)!=8 or {x["case"] for x in a}!=set(old):
        raise ValueError("H133 models not evaluated on the exact same scenes")
    specialist_only=sum(bool(x[metric]) and not bool(old[x["case"]]) for x in a)
    showui_only=sum(not bool(x[metric]) and bool(old[x["case"]]) for x in a)
    both=sum(bool(x[metric]) and bool(old[x["case"]]) for x in a)
    neither=8-specialist_only-showui_only-both
    return {
        "specialist_only":specialist_only,
        "showui_only":showui_only,
        "both":both,"neither":neither,
        "paired_sample_n":8,
        "exact_mcnemar_two_sided_p":exact_discordant_p(specialist_only,showui_only),
        "generalized_superiority_proven":False,
    }
