
# H425: real image-conditioned LoRA, explicitly starts from verified H417
# rather than the genuinely COMPLETE but source-regressed H419 final model.
H425_SEED=42519
H425_OLD_TRAIN=11952
H425_OLD_EN=10976
H425_OLD_AR=976
H425_NEW_PAIR_ACTIONS=32000
H425_SCREENS=16000
H425_AR_TOTAL_PASSES=8
H425_SAMPLES=H425_NEW_PAIR_ACTIONS+H425_OLD_EN+H425_OLD_AR*H425_AR_TOTAL_PASSES
H425_STEPS=H425_SAMPLES//GRAD_ACCUM
H425_UNIQUE_TRAIN_SCREENS=H425_SCREENS+H425_OLD_TRAIN
H425_OUT=Path("/kaggle/working/h425-owner-private-h417-replay-paired-native")
H425_SCHEMA="hala.h425.original-h417-replay-two-prompt-lora.v1"
H425_DONE="COMPLETE_PRIVATE_H425_H417_REPLAY_TRUE_NATIVE_PAIRED"
H425_FAIL="FAILED_PRIVATE_H425_H417_REPLAY_NO_RELEASE"
assert (H425_SAMPLES,H425_STEPS,H425_UNIQUE_TRAIN_SCREENS)==(50784,12696,27952)

def h425_curriculum(train):
    from collections import Counter
    if not isinstance(train,list) or len(train)!=H425_NEW_PAIR_ACTIONS+H425_OLD_TRAIN:
        raise ValueError("H425 original paired and old H417 source rows incomplete")
    pairs={}
    en=[]
    ar=[]
    for idx,r in enumerate(train):
        image=r.get("image_sha256")
        if type(image) is not str or len(image)!=64:
            raise ValueError("H425 source image SHA missing")
        if idx<H425_NEW_PAIR_ACTIONS:
            if r.get("kind")!="original" or r.get("language")!="en":
                raise ValueError("H425 new paired source is not genuine English original")
            pairs.setdefault(image,[]).append(idx)
        elif r.get("language")=="en":
            en.append(idx)
        elif r.get("language")=="ar" and r.get("kind")=="firstparty":
            ar.append(idx)
        else:
            raise ValueError("H425 original H417 source language/kind mismatch")
    if (len(pairs),len(en),len(ar))!=(H425_SCREENS,H425_OLD_EN,H425_OLD_AR):
        raise ValueError("H425 source class totals shifted")
    if set(pairs)&{train[i]["image_sha256"] for i in en+ar}:
        raise ValueError("H425 old H417 TRAIN overlaps H422 original paired screenshot")
    rgen=random.Random(H425_SEED)
    pairs_sorted=[]
    for image in sorted(pairs):
        ids=pairs[image]
        if len(ids)!=2:
            raise ValueError("H425 each original paired screenshot needs TWO actions")
        a,b=(train[i] for i in ids)
        if (a["body"]==b["body"]
            or (a["start"],a["end"])==(b["start"],b["end"])
            or a["domain"]!=b["domain"] or a["id"]==b["id"]):
            raise ValueError("H425 paired original instructions must actually differ")
        pairs_sorted.append(tuple(ids))
    rgen.shuffle(pairs_sorted)
    steps=[tuple(pairs_sorted[i]+pairs_sorted[i+1])
           for i in range(0,len(pairs_sorted),2)]
    rgen.shuffle(en)
    steps.extend(tuple(en[i:i+GRAD_ACCUM]) for i in range(0,len(en),GRAD_ACCUM))
    if len(en)%GRAD_ACCUM:
        raise ValueError("H425 original H417 English replay must group exactly")
    # EIGHT genuine Arabic passes on source-original 976 unique screenshots,
    # with no two same-image AR screenshots in one gradient accumulation.
    for _ in range(H425_AR_TOTAL_PASSES):
        rgen.shuffle(ar)
        steps.extend(tuple(ar[i:i+GRAD_ACCUM]) for i in range(0,len(ar),GRAD_ACCUM))
    if len(steps)!=H425_STEPS or any(len(s)!=GRAD_ACCUM for s in steps):
        raise ValueError("H425 actual 12696-step optimizer curriculum incomplete")
    rgen.shuffle(steps)
    flat=[i for group in steps for i in group]
    counted=Counter(flat)
    expected={i: (H425_AR_TOTAL_PASSES if train[i]["language"]=="ar" else 1)
              for i in range(len(train))}
    if len(flat)!=H425_SAMPLES or counted!=expected:
        raise ValueError("H425 actual original source frequency drift")
    for step in steps:
        members=[train[i] for i in step]
        shas=Counter(r["image_sha256"] for r in members)
        if all(i<H425_NEW_PAIR_ACTIONS for i in step):
            if sorted(shas.values())!=[2,2]:
                raise ValueError("H425 two original JPEGs and two instructions per optimizer step lost")
        elif len(shas)!=4 or any(i<H425_NEW_PAIR_ACTIONS for i in step):
            raise ValueError("H425 source-H417 replay step contains duplicate screenshot or new paired leakage")
    return flat

def h425_prepare_source():
    """Read exact owner H60 source once; freeze old H417 + new H419 DEV."""
    zips,source,cur,fp=get_research_sources()
    extra=[]
    try:
        old_train,old_dev,old_order=partition(source,cur,fp)
        old_train_sha={r["image_sha256"] for r in old_train}
        old_dev_sha={r["image_sha256"] for r in old_dev}
        if (len(old_train),len(old_dev),len(old_order))!=(11952,144,14880):
            raise ValueError("H425 original H417 parent training and DEV drift")
        extra=open_extra_sources(zips)
        new_original=new_verified_images(old_train_sha,old_dev_sha)
        new_source_sha={r["image_sha256"] for r in new_original}
        fresh_dev=new_source_development(
            zips,old_train_sha|old_dev_sha|new_source_sha)
        old_ar=[r for r in old_train if r["language"]=="ar"]
        if (len(new_original),len(new_source_sha),len(old_ar),len(fresh_dev))!=(
            30000,30000,976,96):
            raise ValueError("H425 true H419 30k original and old AR source drift")
        dev=old_dev+fresh_dev
        h419train=new_original+old_ar
        old_sha=sha_bytes("".join(sorted(r["image_sha256"] for r in old_dev)).encode())
        new_sha=sha_bytes("".join(sorted(r["image_sha256"] for r in fresh_dev)).encode())
        if old_sha!="5813a0b8d8f13f96a873c206b532dfb26ed8df65ef385f3a1a6c4e89742e8421" or new_sha!="cd8c25af738b47007f54e606235583f6b59263ef206d0d547ad353e9ec5dde09":
            raise ValueError("H425 frozen original H419 baseline DEV fingerprints drift")
        pairs,unseen=paired_source_rows(h419train,dev)
        if len(pairs)!=H425_NEW_PAIR_ACTIONS:
            raise ValueError("H425 genuine 32k original paired action rows missing")
        train=pairs+old_train
        train_sha={r["image_sha256"] for r in train}
        dev_sha={r["image_sha256"] for r in dev}
        if (len(train)!=43952 or len(train_sha)!=H425_UNIQUE_TRAIN_SCREENS
            or len(dev_sha)!=240 or train_sha&dev_sha or train_sha&unseen):
            raise ValueError("H425 original TRAIN, old/new DEV, H421 counterfactual heldout overlap")
        order=h425_curriculum(train)
        return zips,extra,train,dev,order
    except BaseException:
        for obj in extra:obj.close()
        for obj in zips.values():
            if isinstance(obj,zipfile.ZipFile):obj.close()
        raise

def run_h425():
    import torch
    st={"schema":H425_SCHEMA,"status":"NOT_STARTED",
        "official_base":BASE,"official_revision":REV,
        "h417_source_parent_sha256":PARENT_ADAPTER_SHA,
        "h417_source_parent_receipt_sha256":PARENT_RECEIPT_SHA,
        "h418_new_original_30k_source_manifest_sha256":H418_RECEIPT_SHA,
        "h422_original_pair_rows_sha256":H422_ROWS_SHA,
        "h422_original_paired_source_receipt_sha256":H422_RECEIPT_SHA,
        "h421_reserved_counterfactual_dev_sha256":H421_DEV_SHA,
        "actual_optimizer_steps":0,"processed_training_examples":0,
        "independent_external_ood":False,"source_rights_review_qualified":False,
        "release_qualified":False,"weights_public":False,
        "replaced_failed_h419_parent":True}
    if H425_OUT.exists():
        raise ValueError("H425 create-only owner private output collision")
    H425_OUT.mkdir(parents=True)
    t0=time.monotonic()
    zips=None
    extra=[]
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
            raise ValueError("H425 training requires actual owner FREE Kaggle T4 CUDA")
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0",
            "accelerate==1.12.0","Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        if torch.cuda.get_device_name(0)!="Tesla T4":
            raise ValueError("H425 source-grounded original owner Tesla T4 required")
        original_parent=parent_model_source()
        zips,extra,train,dev,order=h425_prepare_source()
        st.update(h425_unique_original_training_screens=H425_UNIQUE_TRAIN_SCREENS,
            h425_genuine_distinct_native_action_rows=len(train),
            h425_original_same_screenshot_two_instructions=H425_SCREENS,
            h425_original_paired_optimizer_steps=8000,
            h425_old_h417_english_replay_examples=H425_OLD_EN,
            h425_old_h417_arabic_replay_unique=H425_OLD_AR,
            h425_old_h417_arabic_passes=H425_AR_TOTAL_PASSES,
            h425_gradients_expected=H425_SAMPLES,
            h425_optimizer_steps_expected=H425_STEPS,
            untouched_old_h417_and_fresh_h419_original_dev_images=len(dev),
            original_nine_H60_source_zip_SHA256=H419_ALL_SOURCE_SHAS)
        torch.manual_seed(H419_SEED)
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        pretrained=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        model=PeftModel.from_pretrained(pretrained,str(original_parent.parent),
                                        is_trainable=True)
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        st["h417_parent_dev"],st["h417_parent_dev_exact"]=evaluate(
            model,processor,dev,zips,"H425_H417_GENUINE_PRETRAINED_PARENT")
        parent_old=sum(bool(r["correct"]) for r in st["h417_parent_dev_exact"][:144])
        parent_fresh=sum(bool(r["correct"]) for r in st["h417_parent_dev_exact"][144:])
        if parent_old!=76 or parent_fresh!=18:
            raise ValueError("H425 actual H417 source parent original240 model replay drift")
        active=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in n for n,p in active):
            raise ValueError("H425 only proven original H417 LoRA tensors may update")
        before={n:p.detach().float().cpu().clone() for n,p in active}
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=False
        opt=torch.optim.AdamW([p for _,p in active],lr=0.000006)
        losses=[]
        st["private_training_checkpoints_sha256"]={}
        for step in range(H425_STEPS):
            model.train()
            opt.zero_grad(set_to_none=True)
            micro_losses=[]
            for micro in range(GRAD_ACCUM):
                row=train[order[step*GRAD_ACCUM+micro]]
                batch,labels=to_inputs(processor,row,zips)
                labels=batch.pop("labels",labels)
                output=model(**batch)
                loss=coordinate_weighted_loss(output,labels,processor)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("H425 genuine H417-replay native loss non-finite")
                (loss/GRAD_ACCUM).backward()
                micro_losses.append(float(loss.detach().cpu()))
                del output,loss,batch,labels
            torch.nn.utils.clip_grad_norm_([p for _,p in active],1.0)
            opt.step()
            losses.append(round(sum(micro_losses)/len(micro_losses),6))
            st["actual_optimizer_steps"]=step+1
            st["processed_training_examples"]=(step+1)*GRAD_ACCUM
            if (step+1)%150==0:
                print("H425_REAL_H417_REPLAY_AND_PAIRED_NATIVE_GRADIENT",
                    step+1,"/",H425_STEPS,"original_examples",
                    st["processed_training_examples"],"weighted_loss",
                    losses[-1],flush=True)
            if (step+1)%1500==0:
                folder=H425_OUT/"private-checkpoints"/("step-"+str(step+1))
                model.save_pretrained(folder,safe_serialization=True)
                snapshot=folder/"adapter_model.safetensors"
                if not snapshot.is_file() or snapshot.stat().st_size<1000:
                    raise ValueError("H425 genuine intermediate LoRA checkpoint bytes absent")
                st["private_training_checkpoints_sha256"][str(step+1)]={
                    "model_weights_sha256":sha_file(snapshot),
                    "model_weights_bytes":snapshot.stat().st_size,
                    "optimizer_step":step+1}
                st["last_private_checkpoint_step"]=step+1
        st["changed_lora_tensors"]=sum(int(bool(
            (p.detach().float().cpu()!=before[n]).any())) for n,p in active)
        if st["changed_lora_tensors"]<1:
            raise ValueError("H425 real source LoRA weights did not train")
        model.gradient_checkpointing_disable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        st["h425_after_dev"],st["h425_after_dev_exact"]=evaluate(
            model,processor,dev,zips,"H425_TRUE_H417_REPLAY_PAIRED_NATIVE")
        out=H425_OUT/"private-h417-replay-paired-native-adapter"
        model.save_pretrained(out,safe_serialization=True)
        safe=out/"adapter_model.safetensors"
        if not safe.is_file() or safe.stat().st_size<1000:
            raise ValueError("H425 real trained native LoRA adapter missing")
        scores=st["h425_after_dev"]
        old=sum(bool(r["correct"]) for r in st["h425_after_dev_exact"][:144])
        fresh=sum(bool(r["correct"]) for r in st["h425_after_dev_exact"][144:])
        valid=sum(bool(r["valid"]) for r in st["h425_after_dev_exact"])
        parent_ar=st["h417_parent_dev"]["ar"]["correct"]
        after_ar=scores["ar"]["correct"]
        gate=bool(old>=72 and fresh>18 and valid==240)
        # Not a weaker criterion; only declare H425 research improvement if
        # H419's original nonregression is met PLUS total and AR preservation.
        better=bool(gate and old+fresh>parent_old+parent_fresh
                    and after_ar>=parent_ar-1)
        st.update(status=H425_DONE,completed_optimizer_steps=H425_STEPS,
            original_dev_old_parent_correct=parent_old,
            original_dev_old_after_correct=old,
            original_dev_fresh_parent_correct=parent_fresh,
            original_dev_fresh_after_correct=fresh,
            original_dev_parent_ar_correct=parent_ar,
            original_dev_after_ar_correct=after_ar,
            source_only_H419_frozen_research_gate=gate,
            source_only_H425_stronger_improvement_gate=better,
            actual_50784_gradient_examples=H425_SAMPLES,
            model_weights_sha256=sha_file(safe),
            model_weights_bytes=safe.stat().st_size,
            weighted_loss_history=losses,
            device_name=torch.cuda.get_device_name(0),
            elapsed_seconds=round(time.monotonic()-t0,2))
    except Exception as exc:
        st["status"]=H425_FAIL
        st["failure_class"]=type(exc).__name__
        raise
    finally:
        for obj in extra:obj.close()
        if zips:
            for obj in zips.values():
                if isinstance(obj,zipfile.ZipFile):obj.close()
        raw=json.dumps(st,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False).encode("utf8")
        (H425_OUT/"receipt.json").write_bytes(raw)
        (H425_OUT/"receipt.sha256").write_text(sha_bytes(raw))
        print("H425_PRIVATE_SOURCE_MODEL_SUMMARY",json.dumps({
            k:v for k,v in st.items() if not k.endswith("_exact") and k not in
            ("weighted_loss_history","original_nine_H60_source_zip_SHA256")},
            sort_keys=True),flush=True)

if __name__=="__main__":
    run_h425()
