"""H426 real pinned Qwen3-VL tokenizer/real original 32K source target preflight.

CPU-only, no trained model, no images loaded, no source labels/answers logged.
This verifies whether *real tokenizer* target spans can use the proposed
source-coordinate-specific loss before any training or approval.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

_loss_path=Path(__file__).with_name("target_aware_loss.py")
_spec=importlib.util.spec_from_file_location("h426_pinned_target_weight",_loss_path)
_loss_mod=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss_mod)
source_coordinate_token_weights=_loss_mod.source_coordinate_token_weights

BASE="Qwen/Qwen3-VL-2B-Instruct"
REV="89644892e4d85e24eaac8bacfd4f463576704203"
H422_ROWS_SHA="b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa"
QWEN_TOKENIZER_JSON_SHA="a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
QWEN_TOKENIZER_CONFIG_SHA="c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399"
# Exact H425 English prompt suffix in h425/kaggle/h425_h417_replay_paired.py.
H425_EN_SUFFIX=(" Return ONLY JSON {type:drag,start_point:[x,y],end_point:[x,y],"
               "ordered:boolean,coordinate_space:normalized_1000};"
               " coordinates are 0-1000. Do not act.")


def normalized_source_answer(row: dict) -> str:
    w,h=row["image_width"],row["image_height"]
    gold=row["native_drag_completion"]
    if (type(w) is not int or type(h) is not int or w<=0 or h<=0
        or type(gold) is not dict or gold.get("type")!="drag"
        or gold.get("coordinate_space")!="source_image_pixels"
        or type(gold.get("ordered")) is not bool):
        raise ValueError("H426 genuine source row geometry or drag schema invalid")
    def norm(name):
        coords=gold[name]
        if (type(coords) is not list or len(coords)!=2
            or any(type(x) is not int for x in coords)
            or not 0<=coords[0]<=w or not 0<=coords[1]<=h):
            raise ValueError("H426 original source native geometry out of bounds")
        return [round(coords[0]/w*1000),round(coords[1]/h*1000)]
    return json.dumps({"type":"drag","start_point":norm("start_point"),
                       "end_point":norm("end_point"),
                       "ordered":gold["ordered"],
                       "coordinate_space":"normalized_1000"},
                      separators=(",",":"),ensure_ascii=False)


def supervised_pieces(tok,answer: str,request: str="synthetic preflight request") -> list[str]:
    """Mirror H425 user-image/assistant-answer chat template without a GPU/PIL."""
    user=[{"role":"user","content":[{"type":"image"},
                                  {"type":"text","text":request}]}]
    assistant={"role":"assistant","content":[{"type":"text","text":answer}]}
    head=tok.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
    full=tok.apply_chat_template(user+[assistant],tokenize=False,
                                 add_generation_prompt=False)
    before=tok(head,add_special_tokens=False)["input_ids"]
    after=tok(full,add_special_tokens=False)["input_ids"]
    if len(after)<=len(before) or after[:len(before)]!=before:
        raise ValueError("H426 actual Qwen tokenizer chat supervised boundary mismatch")
    return [tok.decode([i],skip_special_tokens=False) for i in after[len(before):]]


def audit(rows_file:Path, *, max_rows:int=0, tokenizer_snapshot:Path|None=None,
          local_only:bool=True) ->dict:
    import transformers
    from transformers import AutoTokenizer
    if transformers.__version__!="4.57.3":
        raise ValueError("H426 training-matched pinned Transformers required")
    raw=rows_file.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=H422_ROWS_SHA:
        raise ValueError("H426 actual original 32K H422 source rows SHA changed")
    if tokenizer_snapshot is not None:
        resolved=Path(tokenizer_snapshot).resolve()
        if (resolved.name!=REV or not (resolved/"tokenizer.json").is_file()
            or not (resolved/"tokenizer_config.json").is_file()):
            raise ValueError("H426 exact pinned tokenizer snapshot files required")
        if (hashlib.sha256((resolved/"tokenizer.json").read_bytes()).hexdigest()
            !=QWEN_TOKENIZER_JSON_SHA or
            hashlib.sha256((resolved/"tokenizer_config.json").read_bytes()).hexdigest()
            !=QWEN_TOKENIZER_CONFIG_SHA):
            raise ValueError("H426 genuine pinned tokenizer source byte digest drift")
        # Loading from the pinned local directory also avoids Transformers'
        # _patch_mistral_regex model_info network request in offline mode.
        tok=AutoTokenizer.from_pretrained(str(resolved),trust_remote_code=False,
                                          local_files_only=True)
    else:
        tok=AutoTokenizer.from_pretrained(BASE,revision=REV,trust_remote_code=False,
                                          local_files_only=local_only)
    counters=collections.defaultdict(collections.Counter)
    total=0
    for line in raw.splitlines():
        if not line:continue
        row=json.loads(line)
        domain=row["source_domain"]
        if (row.get("language")!="en" or type(row.get("instruction")) is not str
            or not 1<=len(row["instruction"])<=4096):
            raise ValueError("H426 original H422 English training instruction invalid")
        answer=normalized_source_answer(row)
        pieces=supervised_pieces(tok,answer,row["instruction"]+"\n"+H425_EN_SUFFIX)
        c=counters[domain]
        c["rows"]+=1
        total+=1
        try:
            mask=source_coordinate_token_weights(pieces,domain)
        except ValueError as e:
            # Never output model predictions, screenshot or original gold targets.
            c["fail_closed"]+=1
            c["fail_reason_"+str(e).split("H426 ")[-1][:65]]+=1
        else:
            c["mask_pass"]+=1
            c["numeric_field_tokens"]+=sum(x>1.0 for x in mask)
            c["supervised_tokens"]+=len(mask)
        if max_rows and total>=max_rows:break
    if total==0:raise ValueError("Original source rows were empty")
    if max_rows==0 and total!=32000:
        raise ValueError("H426 frozen complete H422 32K source cardinality mismatch")
    rejected=sum(c.get("fail_closed",0) for c in counters.values())
    if rejected:
        # No raw examples/targets leak from a failed tokenizer field test.
        raise ValueError(f"H426 original tokenizer masks rejected {rejected}/{total} training actions")
    return {
        "schema":"hala.h426.actual-pinned-qwen-tokenizer-preflight.v1",
        "model":BASE,
        "model_revision":REV,
        "transformers_version":transformers.__version__,
        "tokenizer_class":type(tok).__name__,
        "authentic_h422_rows_sha256":digest,
        "checked_original_action_rows":total,
        "by_domain":{k:dict(v) for k,v in sorted(counters.items())},
        "any_training_executed":False,
        "any_gpu_used":False,
        "independent_external_ood":False,
        "release_qualified":False,
        "no_private_raw_examples_returned":True,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True,type=Path)
    p.add_argument("--max-rows",type=int,default=0)
    p.add_argument("--allow-download",action="store_true")
    p.add_argument("--tokenizer-snapshot",type=Path)
    args=p.parse_args()
    if args.max_rows<0:raise ValueError("Negative source rows preflight count")
    print(json.dumps(audit(args.rows,max_rows=args.max_rows,
                           tokenizer_snapshot=args.tokenizer_snapshot,
                           local_only=not args.allow_download),sort_keys=True))
