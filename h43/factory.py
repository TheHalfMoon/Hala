"""H43 original rights-cleared bilingual state-task pilot. No native UI or imagery.

Shared 3-step workflow: 24 synthetic tasks across 6 app families. Four families
are authoring-only, two are public DEV-only; NEVER call this an independent test.
"""
from __future__ import annotations
import hashlib
import json
import unicodedata

SCHEMA="hala.h43.state-task-pilot.v1"
RIGHTS={"source":"new_first_party_synthetic","license":"CC0-1.0",
        "contains_personal_data":False,"benchmark_trajectories":False,
        "image_training_qualified":False,"native_actions":False}
FAMILIES=(
    ("atlas_supply","Atlas supply board","لوحة مستلزمات أطلس","authoring","list"),
    ("cedar_rooms","Cedar room planner","مخطط قاعات سيدار","authoring","grid"),
    ("orbit_lending","Orbit item lending","إعارة مقتنيات أوربت","authoring","table"),
    ("delta_staffing","Delta shift roster","جدول مناوبات دلتا","authoring","sidebar"),
    ("harbor_inventory","Harbor inventory","مخزون هاربر","dev","grid"),
    ("summit_scheduling","Summit session planner","مخطط جلسات سامت","dev","table"),
)


def canonical(value:object)->bytes:
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")


def digest(value:object)->str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _build()->dict:
    cases=[]
    for family,english,arabic,split,layout in FAMILIES:
        for i in range(4):
            lang="en" if i%2==0 else "ar"
            ar=lang=="ar"
            item=f"{family.upper()}-{i+1:02d}"
            quantity=2+i
            app=arabic if ar else english
            goal=(f"In {app}, select item {item}, set quantity to {quantity}, "
                  "confirm without changing other items." if not ar else
                  f"في {app}، اختر العنصر {item}، واضبط الكمية على {quantity}، "
                  "ثم أكّد دون تعديل العناصر الأخرى.")
            names=[
                f"{item} — "+("Select" if not ar else "اختيار"),
                f"{family.upper()}-OTHER — "+("Select" if not ar else "اختيار"),
                "Quantity" if not ar else "الكمية",
                "Confirm" if not ar else "تأكيد",
                "Change other item" if not ar else "تعديل عنصر آخر",
            ]
            controls=[{"ref":ref,"name":name,"role":"Edit" if ref=="amount_field" else "Button"}
                      for ref,name in zip(
                          ("row_target","row_decoy","amount_field","confirm_button","unrelated_button"),names)]
            cases.append({
                "id":f"h43-{family}-{i:02d}-{lang}",
                "family":family,"split":split,"language":lang,"rtl":ar,
                "app":app,"layout":layout,"goal":goal,
                "target_item":item,"requested_amount":quantity,
                "initial_state":{"selected":None,"amount":None,"confirmed":False,"unrelated_changed":False},
                "desired_state":{"selected":item,"amount":quantity,
                                 "confirmed":True,"unrelated_changed":False},
                "controls":controls,
                "golden_actions":[{"kind":"press","ref":"row_target","value":None},
                                  {"kind":"fill","ref":"amount_field","value":quantity},
                                  {"kind":"press","ref":"confirm_button","value":None}],
                "max_actions":5,"provenance":dict(RIGHTS),"image":None,
            })
    return {"schema":SCHEMA,"benchmark_status":"not_a_benchmark",
            "holdout_status":"not_independent_or_blind",
            "source_revision":"h43-pilot-2026-10-20","cases":cases}


def author_cases()->dict:
    result=_build()
    validate(result)
    return result


def validate(manifest:object)->None:
    if type(manifest) is not dict or set(manifest)!={"schema","benchmark_status","holdout_status",
                                                    "source_revision","cases"}:
        raise ValueError("Malformed H43 task manifest")
    if (manifest["schema"]!=SCHEMA or manifest["benchmark_status"]!="not_a_benchmark"
        or manifest["holdout_status"]!="not_independent_or_blind"
        or manifest["source_revision"]!="h43-pilot-2026-10-20"):
        raise ValueError("Unqualified research cannot claim external or blind benchmark")
    rows=manifest["cases"]
    if type(rows) is not list or len(rows)!=24:raise ValueError("Expected all 24 source cases")
    expected=_build()["cases"]
    families={"authoring":set(),"dev":set()}
    for i,(row,origin) in enumerate(zip(rows,expected)):
        if type(row) is not dict or set(row)!=set(origin):
            raise ValueError("Unknown or missing task fields")
        if row["provenance"]!=RIGHTS or row["image"] is not None:
            raise ValueError("Unlicensed/benchmark/personal/visual data")
        if row["family"] not in {f[0] for f in FAMILIES}:
            raise ValueError("Unknown family")
        if row["language"] not in ("en","ar") or type(row["rtl"]) is not bool:
            raise ValueError("Unsupported language/RTL")
        families[row["split"]].add(row["family"]) if row["split"] in families else (_ for _ in ()).throw(ValueError("Invalid split"))
        if (type(row["goal"]) is not str or row["target_item"] not in row["goal"] or
            str(row["requested_amount"]) not in row["goal"] or
            any(unicodedata.category(c) in ("Cc","Cf","Cs") for c in row["goal"])):
            raise ValueError("Malformed task goal")
        if row != origin:
            raise ValueError("Case differs from independently reauthored canonical source")
    if (families["authoring"] & families["dev"] or len(families["authoring"])!=4
        or len(families["dev"])!=2):
        raise ValueError("Train/DEV family overlap or missing split")


def training_projection(manifest:dict)->list[dict]:
    validate(manifest)
    return [{"id":r["id"],"family":r["family"],"language":r["language"],
             "goal":r["goal"],"controls":r["controls"],"labelled_actions":r["golden_actions"],
             "provenance":r["provenance"],"image":None,
             "visual_training_qualified":False}
            for r in manifest["cases"] if r["split"]=="authoring"]
