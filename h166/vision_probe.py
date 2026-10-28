"""H166 bounded synthetic-only one-step visual merger LoRA feasibility probe.

NO original app gold, no consumed OOD/DEV media, no model checkpoint output.
Memory/gradient feasibility only; does not establish learned task performance.
"""
import json
import math
import os
from pathlib import Path

MAX_PEAK_GIB=13.0
RANK=4
ALPHA=8
EXPECTED_TRANSFORMERS="5.17.0"
EXPECTED_PEFT="0.21.2"
EXPECTED_TORCH="2.11.0"

def visual_merger_targets(named_modules, linear_type=None):
    """Allowlist ONLY Qwen3.5's visual merger linear projections."""
    if linear_type is None:
        import torch
        linear_type=torch.nn.Linear
    matches=[]
    for name,module in named_modules:
        bits=name.split(".")
        if ("visual" in bits and "merger" in bits and
            "language_model" not in bits and isinstance(module,linear_type)):
            matches.append(name)
    if not 1<=len(matches)<=4 or len(set(matches))!=len(matches):
        raise ValueError("H166 visual merger projection signature not qualified")
    if any(".visual.merger." not in ("."+s+".") for s in matches):
        raise ValueError("H166 target must sit inside visual.merger")
    return sorted(matches)

def require_free_t4(torch, installed):
    if installed!={"torch":EXPECTED_TORCH,
                    "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,
                    "Pillow":"11.3.0"}:
        raise ValueError("H166 exact pinned free GPU environment required")
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:
        raise ValueError("H166 free Nvidia T4 CUDA device required")
    name=torch.cuda.get_device_name(0)
    if "T4" not in name or any("T4" not in torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())):
        raise ValueError("H166 paid or unexpected GPU forbidden")
    return name

def synth():
    """Publicly reproducible synthetic red-square example, never heldout gold."""
    from PIL import Image,ImageDraw
    canvas=Image.new("RGB",(960,720),"white")
    ImageDraw.Draw(canvas).rectangle((432,312,528,408),fill=(255,0,0))
    row={"split":"TRAIN","action_type":"click",
         "instruction":"On this artificial white canvas, click the center of the red square.",
         "training_supervision":{"actions":[{"kind":"click","point":[480,360]}]}}
    return canvas,row

def run(output):
    import importlib.metadata
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from h132.grounder_contract import MODEL_ID,MODEL_SHA
    from h164.train_gpu import assemble

    path=Path(output)
    if path.exists() or not str(path).startswith("/kaggle/working/"):
        raise ValueError("H166 private immutable Kaggle output only")
    installed={x:importlib.metadata.version(x).split("+")[0]
               for x in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,installed)
    if os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("H166 Kaggle owner-private research scope missing")
    torch.manual_seed(166)
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    targets=visual_merger_targets(base.named_modules())
    config=LoraConfig(r=RANK,lora_alpha=ALPHA,lora_dropout=0.0,
                      target_modules=targets,bias="none")
    model=get_peft_model(base,config)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in n or ".visual.merger." not in ("."+n+".")
                             for n,p in trainable)):
        raise ValueError("H166 language layers or encoder base weights accidentally trainable")
    opt=torch.optim.AdamW([p for _,p in trainable],lr=1e-5,weight_decay=0)
    img,row=synth()
    batch=assemble(processor,img,row,"cuda:0")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt.zero_grad(set_to_none=True)
    result=model(**batch)
    raw_loss=result.loss
    if raw_loss is None or not torch.isfinite(raw_loss):
        raise ValueError("H166 synthetic visual gradient loss invalid")
    raw_loss.backward()
    grad_count=sum(int(p.grad is not None and
                       torch.isfinite(p.grad).all().item() and
                       torch.count_nonzero(p.grad).item()>0)
                   for _,p in trainable)
    if grad_count<1:
        raise ValueError("H166 no nonzero finite merger gradients")
    before=[p.detach().clone() for _,p in trainable]
    opt.step()
    torch.cuda.synchronize(0)
    changed=sum(int(not torch.equal(old,p.detach()))
                for old,(_,p) in zip(before,trainable))
    peak_alloc=torch.cuda.max_memory_allocated(0)/2**30
    peak_reserved=torch.cuda.max_memory_reserved(0)/2**30
    if not math.isfinite(peak_alloc) or not 0<peak_alloc<=MAX_PEAK_GIB or changed<1:
        raise ValueError("H166 T4 memory cap or trainable visual update failed")
    states=sum(1 for _,p in trainable if opt.state.get(p))
    if states!=len(trainable):
        raise ValueError("H166 optimizer state incomplete")
    path.mkdir(parents=True,exist_ok=False)
    summary={"schema":"hala.h166.synthetic-visual-merger-t4-probe.v1",
       "status":"H166_ONE_SYNTHETIC_VISUAL_MERGER_STEP_PASSED",
       "model_id":MODEL_ID,"model_revision":MODEL_SHA,
       "environment":installed,"gpu":gpu,
       "synthetic_only":True,"source_original_app_images":0,
       "owner_gold_used":False,"new_ood_evaluations":0,"h140_blind_calls":0,
       "optimizer_steps":1,"lora_rank":RANK,"visual_merger_modules":targets,
       "trainable_parameters":sum(p.numel() for _,p in trainable),
       "nonzero_gradient_tensors":grad_count,
       "updated_parameter_tensors":changed,"optimizer_states":states,
       "peak_allocated_gib":round(peak_alloc,3),
       "peak_reserved_gib":round(peak_reserved,3),
       "memory_cap_gib":MAX_PEAK_GIB,
       "finite_synthetic_loss":True,
       "model_weights_saved":False,"paid_GPU":False,"flagship_35B_unchanged":True}
    (path/"h166-visual-memory-receipt.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return summary
