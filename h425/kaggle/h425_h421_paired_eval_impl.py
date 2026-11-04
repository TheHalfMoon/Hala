"""Private H425 vs H417 on true original H421 192 SAME-JPEG/2-prompt DEV.

EVALUATION ONLY after actual H425 TRAIN and source-only pass. This does not
create independent external OOD, release, publish weights or act on a GUI.
"""
H425_CHILD_RECEIPT_FILE="h425_original_model_receipt.json"
H425_CHILD_RECEIPT_SHA_FILE="h425_original_model_receipt.sha256"
H425_CHILD_MODEL_FILE="h425_original_trained_adapter.safetensors"
H425_CHILD_MODEL_CONFIG="h425_original_trained_adapter_config.json"
H425_EXPECTED_STEPS=12696
H425_EXPECTED_GRADIENT_SAMPLES=50784
H425_TRUTH_H422_ROWS_SHA="b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa"
H425_TRUTH_H422_SOURCE_RECEIPT_SHA="229097839788875059607e4934418c3c940ef32620c46083b38261aae061cf87"
H425_EXPECTED_SCHEMA="hala.h425.original-h417-replay-two-prompt-lora.v1"
H425_EXPECTED_STATUS="COMPLETE_PRIVATE_H425_H417_REPLAY_TRUE_NATIVE_PAIRED"
H425_H421_OUTPUT=Path("/kaggle/working/h425-owner-private-h421-counterfactual-model-eval")

def h425_true_trained_child():
    """Accept only actual self-consistent COMPLETED + nonregressed H425 LoRA."""
    remote=only_input_file(H425_CHILD_RECEIPT_FILE)
    sidecar=only_input_file(H425_CHILD_RECEIPT_SHA_FILE)
    digest=sha_file(remote)
    if sidecar.read_text("ascii").strip()!=digest:
        raise ValueError("H425 original trained model receipt SHA sidecar mismatch")
    r=strict_load(remote.read_text("utf8"))
    if (r.get("schema")!=H425_EXPECTED_SCHEMA
        or r.get("status")!=H425_EXPECTED_STATUS
        or r.get("official_base")!=BASE or r.get("official_revision")!=REV
        or r.get("h417_source_parent_sha256")!=PARENT_ADAPTER_SHA
        or r.get("h417_source_parent_receipt_sha256")!=PARENT_RECEIPT_SHA
        or r.get("h418_new_original_30k_source_manifest_sha256")!=H418_RECEIPT_SHA
        or r.get("h422_original_pair_rows_sha256")!=H425_TRUTH_H422_ROWS_SHA
        or r.get("h422_original_paired_source_receipt_sha256")!=H425_TRUTH_H422_SOURCE_RECEIPT_SHA
        or r.get("h421_reserved_counterfactual_dev_sha256")!=SOURCE_JSONL_SHA
        or r.get("original_nine_H60_source_zip_SHA256")!=H419_ALL_SOURCE_SHAS
        or r.get("actual_optimizer_steps")!=H425_EXPECTED_STEPS
        or r.get("completed_optimizer_steps")!=H425_EXPECTED_STEPS
        or r.get("processed_training_examples")!=H425_EXPECTED_GRADIENT_SAMPLES
        or r.get("h425_unique_original_training_screens")!=27952
        or r.get("h425_genuine_distinct_native_action_rows")!=43952
        or r.get("h425_original_same_screenshot_two_instructions")!=16000
        or r.get("h425_old_h417_english_replay_examples")!=10976
        or r.get("h425_old_h417_arabic_replay_unique")!=976
        or r.get("h425_old_h417_arabic_passes")!=8
        or r.get("source_only_H419_frozen_research_gate") is not True
        or r.get("source_only_H425_stronger_improvement_gate") is not True
        or r.get("independent_external_ood") is not False
        or r.get("source_rights_review_qualified") is not False
        or r.get("release_qualified") is not False
        or r.get("weights_public") is not False
        or type(r.get("changed_lora_tensors")) is not int
        or r["changed_lora_tensors"]<1):
        raise ValueError("H425 not genuinely source-qualified for the original H421 counterfactual DEV")
    a,b=r.get("h417_parent_dev_exact"),r.get("h425_after_dev_exact")
    if type(a) is not list or type(b) is not list or len(a)!=240 or len(b)!=240:
        raise ValueError("H425 original 240-case parent and trained model predictions missing")
    for x,y in zip(a,b):
        if (type(x) is not dict or type(y) is not dict
            or any(x.get(z)!=y.get(z) for z in (
                "id","image_sha256","domain","family","language"))
            or type(x.get("correct")) is not bool
            or type(y.get("correct")) is not bool
            or type(y.get("valid")) is not bool):
            raise ValueError("H425 originally source-paired model case proof incomplete")
    old_parent=sum(int(x["correct"]) for x in a[:144])
    fresh_parent=sum(int(x["correct"]) for x in a[144:])
    old_new=sum(int(x["correct"]) for x in b[:144])
    fresh_new=sum(int(x["correct"]) for x in b[144:])
    parent_ar=sum(int(x["correct"]) for x in a if x["language"]=="ar")
    after_ar=sum(int(x["correct"]) for x in b if x["language"]=="ar")
    if ((old_parent,fresh_parent)!=(76,18) or old_new<72 or fresh_new<=18
        or old_new+fresh_new<=94 or after_ar<parent_ar-1
        or sum(int(x["valid"]) for x in b)!=240
        or r.get("original_dev_old_parent_correct")!=old_parent
        or r.get("original_dev_fresh_parent_correct")!=fresh_parent
        or r.get("original_dev_old_after_correct")!=old_new
        or r.get("original_dev_fresh_after_correct")!=fresh_new
        or r.get("original_dev_parent_ar_correct")!=parent_ar
        or r.get("original_dev_after_ar_correct")!=after_ar
        or r.get("h417_parent_dev",{}).get("all",{}).get("correct")!=old_parent+fresh_parent
        or r.get("h425_after_dev",{}).get("all",{}).get("correct")!=old_new+fresh_new):
        raise ValueError("H425 actual original 240 DEV/Arabic did not independently reproduce frozen gate")
    weights=only_input_file(H425_CHILD_MODEL_FILE)
    if (sha_file(weights)!=r.get("model_weights_sha256")
        or weights.stat().st_size!=r.get("model_weights_bytes")
        or weights.stat().st_size<=1000):
        raise ValueError("H425 actual private trained LoRA safetensors SHA mismatch")
    cfg=only_input_file(H425_CHILD_MODEL_CONFIG)
    if cfg.stat().st_size<100:
        raise ValueError("H425 private model LoRA adapter config missing")
    parsed=strict_load(cfg.read_text("utf8"))
    if (parsed.get("base_model_name_or_path")!=BASE
        or parsed.get("peft_type")!="LORA"
        or parsed.get("r")!=8 or parsed.get("lora_alpha")!=16):
        raise ValueError("H425 trained LoRA original model config invalid")
    return r,digest,weights,cfg

def h425_true_h421_pair_sources():
    """Frozen genuinely NEW H421 JPEGs also excluded from H425 old H417 TRAIN."""
    pairs,flat,cases,zips,extra=h421_verified_original_rows()
    zips_old=None
    # h421_verified_original_rows proves disjoint from H419 new30K +
    # Arabic and source-old DEV. Explicitly check H417 old English TRAIN,
    # which H425 newly replayed (H419 had not replayed it).
    try:
        zips_old,source,curr,fp=get_research_sources()
        old_train,old_dev,_=partition(source,curr,fp)
        old_sha={r["image_sha256"] for r in old_train}
        eval_sha={r["image_sha256"] for r in cases}
        if (len(eval_sha)!=IMAGE_PAIRS or len(old_train)!=11952
            or old_sha & eval_sha):
            raise ValueError("H425 H421 original two-instruction DEV overlaps old H417 replay TRAIN")
        return pairs,flat,cases,zips,extra
    except BaseException:
        for obj in extra:obj.close()
        for obj in zips.values():
            if isinstance(obj,zipfile.ZipFile):obj.close()
        raise
    finally:
        if zips_old:
            for z in zips_old.values():
                if isinstance(z,zipfile.ZipFile):z.close()

def run_h425_h421():
    import torch
    if H425_H421_OUTPUT.exists():
        raise ValueError("H425 H421 genuine original model comparison output create-only")
    H425_H421_OUTPUT.mkdir(parents=True)
    st={"schema":"hala.h425.original-paired-h421-384-model-eval.v1",
        "status":"NOT_STARTED","official_base":BASE,"official_revision":REV,
        "h417_parent_adapter_sha256":PARENT_ADAPTER_SHA,
        "source_pairs_jsonl_sha256":SOURCE_JSONL_SHA,
        "source_pairs_receipt_sha256":SOURCE_RECEIPT_SHA,
        "model_training_executed_in_this_kernel":False,
        "independent_external_ood":False,"model_release_qualified":False,
        "weights_public":False}
    zips=None;extra=[]
    started=time.monotonic()
    temp_dir=Path("/kaggle/working/.h425-temp-qualified-actual-child")
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
            raise ValueError("H425 true original H421 model eval requires owner FREE private T4")
        if torch.cuda.get_device_name(0)!="Tesla T4":
            raise ValueError("H425 original unseen paired evaluation demands genuine Tesla T4")
        subprocess.run([sys.executable,"-m","pip","install","-q",
           "transformers==4.57.3","peft==0.18.0","accelerate==1.12.0",
           "Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        parent_path=parent_model_source()
        receipt,receipt_sha,weight,cfg=h425_true_trained_child()
        frozen,flat,cases,zips,extra=h425_true_h421_pair_sources()
        st.update(h425_completed_source_optimizer_steps=receipt["actual_optimizer_steps"],
                  h425_source_only_research_gate_passed=True,
                  h425_authentic_trained_adapter_sha256=sha_file(weight),
                  h425_authentic_private_receipt_sha256=receipt_sha,
                  original_heldout_images=IMAGE_PAIRS,
                  native_contrastive_original_instructions=SAMPLE_N,
                  device_name=torch.cuda.get_device_name(0))
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        for label,path in (("H417_PARENT",parent_path.parent),("H425_TRAINED",None)):
            if path is None:
                temp_dir.mkdir(exist_ok=False)
                shutil.copyfile(weight,temp_dir/"adapter_model.safetensors")
                shutil.copyfile(cfg,temp_dir/"adapter_config.json")
                if sha_file(temp_dir/"adapter_model.safetensors")!=receipt["model_weights_sha256"]:
                    raise ValueError("H425 staged exact real LoRA trained bytes changed")
                path=temp_dir
            base=AutoModelForImageTextToText.from_pretrained(
                BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
                attn_implementation="sdpa").to("cuda:0")
            net=PeftModel.from_pretrained(base,str(path),is_trainable=False)
            preds=h421_generate(net,processor,cases,zips,label)
            if len(preds)!=SAMPLE_N:
                raise ValueError("H425 genuinely heldout 384 model outputs incomplete")
            stats=calculate(flat,preds)
            st[label.lower()+"_pair_stats"]=stats
            st[label.lower()+"_exact_predictions"]=preds
            del net,base,preds
            gc.collect();torch.cuda.empty_cache()
            if path==temp_dir:shutil.rmtree(temp_dir)
        st.update(status="COMPLETE_PRIVATE_H417_VS_H425_TRUE_ORIGINAL_PAIRED",
            research_pair_improved=bool(
                st["h425_trained_pair_stats"]["both_actions_correct"]>
                st["h417_parent_pair_stats"]["both_actions_correct"]),
            duration_seconds=round(time.monotonic()-started,2))
    except Exception as exc:
        st.update(status="FAILED_PRIVATE_H425_H421_MODEL_EVAL_NO_RELEASE",
                  failure_class=type(exc).__name__)
        raise
    finally:
        if temp_dir.exists():shutil.rmtree(temp_dir)
        for obj in extra:obj.close()
        if zips:
            for obj in zips.values():
                if isinstance(obj,zipfile.ZipFile):obj.close()
        raw=json.dumps(st,sort_keys=True,indent=2,allow_nan=False,
                       ensure_ascii=False).encode()
        (H425_H421_OUTPUT/"receipt.json").write_bytes(raw)
        (H425_H421_OUTPUT/"receipt.sha256").write_text(sha_bytes(raw))
        print("H425_H421_REAL_PAIR_MODEL_EVAL_PRIVATE_RESULT",
              json.dumps({k:v for k,v in st.items()
                          if not k.endswith("_exact_predictions")},sort_keys=True),
              flush=True)

if __name__=="__main__":
    run_h425_h421()
