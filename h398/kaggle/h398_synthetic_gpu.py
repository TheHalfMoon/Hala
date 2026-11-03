"""H398 private GPU research: 100% generated, original bilingual grid grounding.

An intentionally TINY conditional visual classifier, NOT Hala 2B/35B.
No scraped source, GUI screenshots, external dataset, model download or H199 bypass.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path

SEED=398
COLORS=((0.91,0.17,0.20),(0.14,0.83,0.25),(0.19,0.29,0.91))
SCHEMA="hala.h398.private-invented-grid-gpu-smoke.v1"

def samples(n,seed,theme="normal"):
    import torch
    rng=random.Random(seed)
    images=torch.zeros((n,3,48,48),dtype=torch.float32)
    tokens=torch.zeros(n,dtype=torch.long)
    targets=torch.zeros(n,dtype=torch.long)
    base=0.10 if theme=="normal" else 0.23
    for i in range(n):
        language_color=i%6
        tokens[i]=language_color
        target_color=language_color%3
        positions=rng.sample(range(9),9)
        targets[i]=positions[target_color]
        images[i].fill_(base)
        for j,p in enumerate(positions):
            row,col=divmod(p,3)
            if j<3:
                color=COLORS[j]
            else:
                grey=0.30+(j-3)*0.045
                color=(grey,grey,grey)
            if theme=="unseen":
                color=tuple(max(0,min(1,c*0.85+0.045)) for c in color)
            y,x=row*16+4,col*16+4
            for channel,value in enumerate(color):
                images[i,channel,y:y+8,x:x+8]=value
    return images,tokens,targets

def net():
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision=nn.Sequential(nn.Conv2d(3,32,3,padding=1),
                nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,48,3,padding=1),
                nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(48,48,3,padding=1),
                nn.ReLU(),nn.AdaptiveAvgPool2d((3,3)))
            self.text=nn.Embedding(6,48)
        def forward(self,images,ids):
            features=self.vision(images)
            vectors=self.text(ids).unsqueeze(-1).unsqueeze(-1)
            return (features*vectors).sum(dim=1).flatten(start_dim=1)
    return Model()

def evaluate(model,corpus,device):
    import torch
    images,tokens,labels=corpus
    model.eval()
    with torch.inference_mode():
        predictions=model(images.to(device),tokens.to(device)).argmax(-1).cpu()
    en=(tokens<3)
    ar=(tokens>=3)
    return {"all":round((predictions==labels).float().mean().item(),4),
            "en":round((predictions[en]==labels[en]).float().mean().item(),4),
            "ar":round((predictions[ar]==labels[ar]).float().mean().item(),4),
            "unique_grid_positions":int(torch.unique(predictions).numel()),
            "n":len(labels)}

def run(steps=160,n_train=960):
    import torch
    if not torch.cuda.is_available() or not any("T4" in torch.cuda.get_device_name(i)
                                                for i in range(torch.cuda.device_count())):
        raise RuntimeError("H398 free Kaggle NVIDIA T4 required")
    if steps!=160 or n_train!=960:
        raise ValueError("H398 frozen small synthetic-only training plan")
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(SEED)
    device="cuda:0"
    # All examples and labels are generated deterministically from our own code.
    train=samples(n_train,39810)
    dev=samples(180,39820)
    unseen=samples(180,39830,theme="unseen")
    # No reuse of exact image+instruction case between distinct source splits.
    def keys(corpus):
        images,tokens,_=corpus
        return {hashlib.sha256(images[i].numpy().tobytes()+bytes([tokens[i].item()])).hexdigest()
                for i in range(len(tokens))}
    train_keys,dev_keys,unseen_keys=map(keys,(train,dev,unseen))
    if (len(train_keys)!=n_train or len(dev_keys)!=180 or len(unseen_keys)!=180
            or train_keys&dev_keys or train_keys&unseen_keys or dev_keys&unseen_keys):
        raise RuntimeError("H398 invented dataset sample identity collision")
    model=net().to(device)
    baseline_dev=evaluate(model,dev,device)
    baseline_unseen=evaluate(model,unseen,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
    generator=torch.Generator().manual_seed(SEED)
    losses=[]
    started=time.monotonic()
    for step in range(steps):
        model.train()
        index=torch.randint(0,n_train,(48,),generator=generator)
        batch=train[0][index].to(device)
        instructions=train[1][index].to(device)
        labels=train[2][index].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch,instructions)
        loss=torch.nn.functional.cross_entropy(logits,labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite H398 synthetic loss")
        loss.backward()
        optimizer.step()
        if step%40==0 or step==steps-1:
            losses.append(round(loss.item(),6))
            print("H398 real original synthetic optimizer step",step+1,"/",steps,
                  "loss",losses[-1],flush=True)
    final_dev=evaluate(model,dev,device)
    final_unseen=evaluate(model,unseen,device)
    receipt={"schema":SCHEMA,"status":"SYNTHETIC_GPU_TRAINED_NOT_HALA_MODEL",
             "source":"algorithmically generated new 48x48 color-grid tasks",
             "languages":["en","ar"],"no_external_gui_screenshots":True,
             "no_h199_rights_bypass":True,"no_private_inputs":True,
             "no_model_download":True,"no_ood_custodian_claim":True,
             "no_flagship_weights":True,"model":"original tiny CNN+instruction embeddings, not Hala/Qwen",
             "device_name":torch.cuda.get_device_name(0),
             "cuda_device_count":torch.cuda.device_count(),
             "train_n":n_train,"dev_n":180,"research_unseen_n":180,
             "optimizer_steps":steps,"loss_checkpoints":losses,
             "baseline_dev":baseline_dev,"trained_dev":final_dev,
             "baseline_unseen":baseline_unseen,"trained_unseen":final_unseen,
             "train_seconds":round(time.monotonic()-started,2),
             "scientific_limits":"not Qwen or Hala model; OOD is author-exposed, not independently sealed"}
    path=Path("/kaggle/working/h398-synthetic-research")
    path.mkdir(parents=True,exist_ok=False)
    encoded=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()
    (path/"receipt.json").write_bytes(encoded)
    (path/"receipt.sha256").write_text(hashlib.sha256(encoded).hexdigest())
    torch.save(model.cpu().state_dict(),path/"original_tiny_synthetic_weights.pt")
    print("H398_RECEIPT",json.dumps(receipt,sort_keys=True),flush=True)
    return receipt

if __name__=="__main__":
    run()
