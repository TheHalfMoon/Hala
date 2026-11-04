
# H424 model checkpoint diagnostics. Source imports/helpers are prepended
# verbatim from the SHA-pinned original h419_native_drag_continuation.py.
H424_SCHEMA="hala.h424.actual-h419-owner-checkpoint-model-diagnostic.v1"
H424_MANIFEST="h424_checkpoint_manifest.json"
H424_ZIP="h424_h419_owner_private_checkpoint_bundle.zip"
H424_MANIFEST_SHA="b9b542b9511e493cb46087acb42ba13982fcde6685049e2e0c885d5a6f91fd8c"
H424_ZIP_SHA="59ff83f85461ae64baa312453af6314560c3fb305907105a8d5a213eaa8c42ee"
H424_H419_RECEIPT_SHA="f8c491d8522a8349974d05318fded7cbe0349478be27efcbb118a8efdcfa37ac"
H424_FINAL_SHA="96d07b07cca54d15e9ce11da4d26fa3748564f09a2974d9a0b48e419ab03eb8c"
H424_CHECKPOINTS=(1500,3000,4500,6000,7500,9000)
H424_OUT=Path("/kaggle/working/h424-owner-private-checkpoint-diagnostic")
H424_TMP=Path("/tmp/h424-original-private-weights")
H424_INPUT=Path("/kaggle/input")

class H424ExtractedArchive:
    """Read Kaggle's automatic ZIP-extraction result as immutable members."""

    def __init__(self, root):
        self.root=Path(root)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def namelist(self):
        # No alternate/extra source members and no symlink-backed model bytes.
        if any(p.is_symlink() for p in self.root.rglob("*")):
            raise ValueError("H424 private extracted dataset contains symlinks")
        return [p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*") if p.is_file()]
    def read(self, name):
        path=self.root/name
        if (not path.is_file() or path.is_symlink()
            or ".." in Path(name).parts
            or not path.resolve().is_relative_to(self.root.resolve())):
            raise ValueError("H424 original extracted source member path invalid")
        return path.read_bytes()

def h424_open_private_archive():
    import zipfile
    zip_files=list(H424_INPUT.rglob(H424_ZIP))
    unpacked=list(H424_INPUT.rglob(Path(H424_ZIP).stem))
    sources=[p for p in zip_files if p.is_file()]+[p for p in unpacked if p.is_dir()]
    if len(sources)!=1:
        raise ValueError("H424 private checkpoint archive absent or ambiguous")
    archive=sources[0]
    if archive.is_file():
        if sha_file(archive)!=H424_ZIP_SHA:
            raise ValueError("H424 original full ZIP archive SHA drift")
        return zipfile.ZipFile(archive)
    # Kaggle auto-expands uploaded ZIPs. Every extracted file must still pass
    # the original per-model SHA and 240-result receipt verification below.
    return H424ExtractedArchive(archive)

def h424_gate(parent_old,old,parent_fresh,fresh,valid,expected=240):
    """Research gate from H419, unchanged. Not external qualification."""
    if any(type(x) is not int or x < 0 for x in
           (parent_old,old,parent_fresh,fresh,valid,expected)):
        raise ValueError("H424 score counts must be nonnegative exact integers")
    if expected!=240 or valid>expected or parent_old>144 or old>144 or parent_fresh>96 or fresh>96:
        raise ValueError("H424 impossible original H419 144-old/96-new DEV scores")
    return valid==expected and fresh>parent_fresh and old>=parent_old-4

def h424_validate_private_bundle():
    """Check original provider output and all 7 immutable adapters BEFORE GPU."""
    import zipfile
    mani=only_input_file(H424_MANIFEST)
    raw=mani.read_bytes()
    if sha_bytes(raw)!=H424_MANIFEST_SHA:
        raise ValueError("H424 owner staged checkpoint manifest SHA drift")
    data=strict_load(raw.decode("utf8"))
    if (data.get("schema")!="hala.h424.actual-owner-h419-checkpoints.v1"
        or data.get("owner_h419_receipt_sha256")!=H424_H419_RECEIPT_SHA
        or data.get("archive_sha256")!=H424_ZIP_SHA
        or data.get("original_completed_optimizer_steps")!=9452
        or data.get("original_model_research_gate") is not False
        or data.get("model_release_qualified") is not False
        or data.get("independent_external_ood") is not False
        or set(data.get("models",{}))!={"final",*(str(x) for x in H424_CHECKPOINTS)}):
        raise ValueError("H424 owner original H419 checkpoint provenance changed")
    with h424_open_private_archive() as z:
        names=set(z.namelist())
        expected={"receipt.json","receipt.sha256"}
        for item in data["models"]:
            expected.add(f"models/{item}/adapter_model.safetensors")
            expected.add(f"models/{item}/adapter_config.json")
        if names!=expected:
            raise ValueError("H424 private checkpoint ZIP unexpectedly added/removed content")
        receipt=z.read("receipt.json")
        if (sha_bytes(receipt)!=H424_H419_RECEIPT_SHA
            or z.read("receipt.sha256").decode("ascii").strip()!=H424_H419_RECEIPT_SHA):
            raise ValueError("H424 completed H419 original trained receipt mismatch")
        run=strict_load(receipt.decode("utf8"))
        if (run.get("schema")!="hala.h419.private-new-30000-original-native-drag.v1"
            or run.get("status")!=H419_PRIVATE_STATUS
            or run.get("official_base")!=BASE
            or run.get("official_revision")!=REV
            or run.get("h417_original_parent_adapter_sha256")!=PARENT_ADAPTER_SHA
            or run.get("h417_original_parent_receipt_sha256")!=PARENT_RECEIPT_SHA
            or run.get("h418_original_source_receipt_sha256")!=H418_RECEIPT_SHA
            or run.get("h418_original_source_rows_sha256")!=H418_ROWS_SHA
            or run.get("actual_optimizer_steps")!=9452
            or run.get("processed_training_examples")!=37808
            or run.get("source_only_research_gate") is not False
            or run.get("model_release_qualified") is not False
            or run.get("adapter_sha256")!=H424_FINAL_SHA):
            raise ValueError("H424 original H419 final model / failed gate changed")
        if H424_TMP.exists():
            raise ValueError("H424 private weights scratch collision")
        H424_TMP.mkdir(parents=True)
        for label,entry in data["models"].items():
            if (type(entry) is not dict
                or entry.get("step")!=(9452 if label=="final" else int(label))
                or entry.get("adapter_bytes")!=12878000):
                raise ValueError("H424 source checkpoint step or bytes changed")
            for filename, digest_key in (
                ("adapter_model.safetensors","adapter_sha256"),
                ("adapter_config.json","config_sha256")):
                b=z.read(f"models/{label}/{filename}")
                if sha_bytes(b)!=entry[digest_key]:
                    raise ValueError("H424 actual source checkpoint bytes differ from manifest")
                target=H424_TMP/label
                target.mkdir(exist_ok=True)
                (target/filename).write_bytes(b)
            config=strict_load((H424_TMP/label/"adapter_config.json").read_text())
            if config.get("base_model_name_or_path") not in (None,BASE):
                raise ValueError("H424 genuine H419 PEFT adapter uses a different base")
        if data["models"]["final"]["adapter_sha256"]!=H424_FINAL_SHA:
            raise ValueError("H424 real final trained adapter SHA mismatch")
    return data,run

def h424_replay_match(records,reference):
    if type(records) is not list or len(records)!=240:
        raise ValueError("H424 original DEV exact 240 model predictions missing")
    for got,ref in zip(records,reference):
        for key in ("id","family","domain","language","image_sha256"):
            if got.get(key)!=ref.get(key):
                raise ValueError("H424 original H419 held-out source DEV identity/order changed")
        if type(got.get("correct")) is not bool or type(got.get("valid")) is not bool:
            raise ValueError("H424 real model output correctness not boolean")

def run_h424():
    import torch
    import gc
    import shutil
    import subprocess
    import sys
    started=time.monotonic()
    status={"schema":H424_SCHEMA,"status":"NOT_STARTED","source_h419_final_gate":False,
            "source_only_selection":True,"independent_external_ood":False,
            "model_release_qualified":False,"weights_public":False,"all_checkpoints_actual_optimizer_steps":[]}
    if H424_OUT.exists():
        raise ValueError("H424 create-only private diagnostic output collision")
    H424_OUT.mkdir(parents=True)
    zips=None
    extras=[]
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
            raise ValueError("H424 owner FREE Kaggle CUDA T4 required")
        manifest,orig=h424_validate_private_bundle()
        if torch.cuda.get_device_name(0)!="Tesla T4":
            raise ValueError("H424 source experiment requires original owner T4")
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0","accelerate==1.12.0",
            "Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        parent_path=parent_model_source()
        zips,extras,train,dev,order,old_dev,new_dev=h419_prepare_sources()
        if (len(train),len(dev),len(old_dev),len(new_dev))!=(30976,240,144,96):
            raise ValueError("H424 exact original 240 DEV partitions changed")
        if (orig.get("old_h417_dev_sha256")!=sha_bytes("".join(
            sorted(c["image_sha256"] for c in old_dev)).encode())
            or orig.get("new_dev_sha256")!=sha_bytes("".join(
            sorted(c["image_sha256"] for c in new_dev)).encode())):
            raise ValueError("H424 original author-source DEV fingerprints changed")
        torch.manual_seed(H419_SEED)
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        pretrained=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        model=PeftModel.from_pretrained(pretrained,str(parent_path.parent),is_trainable=False)
        status.update(original_source_image_disjoint_dev_n=len(dev),
            qualified_owner_receipt_sha256=H424_H419_RECEIPT_SHA,
            private_manifest_sha256=H424_MANIFEST_SHA,
            source_zip_sha256=H419_ALL_SOURCE_SHAS,
            checkpoint_adapter_sha256={n:d["adapter_sha256"] for n,d in manifest["models"].items()},
            final_H419_parent_old_correct=orig["old_h417_dev_parent_correct"],
            final_H419_parent_fresh_correct=orig["fresh_new_dev_parent_correct"])
        observed={}
        # Final original 9452-step model evaluated FIRST to prove this environment
        # reproduces the 240-image H419 receipt before any model selection.
        for label in ("final",*(str(i) for i in H424_CHECKPOINTS)):
            folder=H424_TMP/label
            name="h424_"+label
            model.load_adapter(str(folder),adapter_name=name,is_trainable=False)
            model.set_adapter(name)
            scores,exact=evaluate(model,processor,dev,zips,"H424_SOURCE_ACTUAL_"+label)
            h424_replay_match(exact,orig["adapted_dev_exact"])
            old=sum(int(x["correct"]) for x in exact[:144])
            fresh=sum(int(x["correct"]) for x in exact[144:])
            valid=sum(int(x["valid"]) for x in exact)
            if label=="final" and (scores!=orig["adapted_dev"] or old!=71 or fresh!=30):
                raise ValueError("H424 final H419 actual adapter failed exact owner model score replay")
            allowed=h424_gate(76,old,18,fresh,valid)
            observed[label]={"step":manifest["models"][label]["step"],
                "adapter_sha256":manifest["models"][label]["adapter_sha256"],
                "old_144_correct":old,"fresh_96_correct":fresh,
                "total_240_correct":old+fresh,"valid_240":valid,
                "source_only_research_gate":allowed,
                "source_exact_predictions":exact}
            status["all_checkpoints_actual_optimizer_steps"].append(manifest["models"][label]["step"])
            print("H424_REAL_CHECKPOINT",label,"old",old,"fresh",fresh,"valid",valid,
                  "research_gate",allowed,flush=True)
        passing=[(name,case) for name,case in observed.items()
                 if case["source_only_research_gate"]]
        passing.sort(key=lambda z:(-z[1]["total_240_correct"],
                  -z[1]["old_144_correct"],z[1]["step"]))
        winner=passing[0][0] if passing else None
        status.update(status="COMPLETE_PRIVATE_ORIGINAL_CHECKPOINT_SCAN",
                      all_7_actual_checkpoint_scores=observed,
                      original_final_gate_reproduced=False,
                      source_only_selected_checkpoint=winner,
                      selected_checkpoint_sha256=observed[winner]["adapter_sha256"] if winner else None,
                      note="DEV-based source-only selection, NOT independently held-out OOD and NOT release",
                      elapsed_seconds=round(time.monotonic()-started,2))
    except Exception as exc:
        status.update(status="FAILED_PRIVATE_CHECKPOINT_SCAN_NO_RELEASE",
                      failure_class=type(exc).__name__)
        raise
    finally:
        if zips:
            for z in zips.values():
                if hasattr(z,"close"):z.close()
        for z in extras:z.close()
        raw=json.dumps(status,sort_keys=True,indent=2,allow_nan=False).encode()
        (H424_OUT/"receipt.json").write_bytes(raw)
        (H424_OUT/"receipt.sha256").write_text(sha_bytes(raw))
        print("H424_PRIVATE_SUMMARY",json.dumps(
            {k:v for k,v in status.items() if k!="all_7_actual_checkpoint_scores"
             and k!="source_zip_sha256"},sort_keys=True),flush=True)
        if H424_TMP.exists():shutil.rmtree(H424_TMP)

if __name__=="__main__":
    run_h424()
