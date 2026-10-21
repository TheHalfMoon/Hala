"""H49 actual remote-only frozen baseline/adapter GUI inference producer.

No execution on personal Windows/Mac; explicitly admitted free Colab CUDA only.
Never reads evaluator_only, capture, training completion, or official benchmark labels.
Retains genuine raw attempts and original error evidence; not a release gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from h49.factory import WIDTH, HEIGHT, sha
from h49.paired_eval import (SCHEMA, MODEL, REVISION, digest_prompt, public_prompt)
from training.artifacts import strict_json

ROOT = Path(__file__).resolve().parent.parent
PINNED = {"torch": "2.11.0", "transformers": "5.16.1",
          "peft": "0.20.0", "Pillow": "12.3.0"}
FIELDS = {"id","family","language","split","goal","image",
          "image_sha256","width","height","independent_blind_test"}


def admit_environment():
    """Fail without actually allocated GPU, explicit noncommercial free-tier consent."""
    if (platform.system() != "Linux" or
        os.environ.get("COLAB_RELEASE_TAG") is None or
        os.environ.get("HALA_H49_FREE_COLAB_RESEARCH") != "1" or
        os.environ.get("HALA_H49_NO_PAID_RUNTIME_CONFIRMED") != "1" or
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists()):
        raise RuntimeError("Only explicitly opted-in free noncommercial Google Colab Linux")
    # Environment variables express the operator's confirmation; they do not
    # independently prove account billing/plan or free resource entitlement.


def source_gate(commit):
    if not isinstance(commit, str) or not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Exact signed published source commit required")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=35)
    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("Check out exact original H49 source commit")
    tracked = ["h49/factory.py","h49/paired_eval.py","h49/remote_infer.py",
               "h49/assets/model_inputs.json","h49/assets/source.json",
               "h49/assets/sft_manifest.json"]
    # Check all real model-facing screenshots, not the privileged evaluator file.
    tracked += ["h49/assets/png/" + c["id"] + ".png" for c in public_cases()]
    digests = {}
    for path in tracked:
        current = (ROOT/path).read_bytes()
        if current != git("show", commit + ":" + path):
            raise ValueError("Source/media differ from exact published commit: " + path)
        digests[path] = sha(current)
    return digests


def public_cases():
    envelope = strict_json((ROOT/"h49/assets/model_inputs.json").read_text("utf-8"))
    cases = envelope.get("cases", [])
    dev = [r for r in cases if isinstance(r, dict) and r.get("split") == "dev"]
    if (envelope.get("schema") != "hala.h49.model-only.v1" or
        envelope.get("independent_blind_test") is not False or
        len(cases) != 96 or len(dev) != 32 or
        sum(r.get("language") == "ar" for r in dev) != 16 or
        sum(r.get("language") == "en" for r in dev) != 16 or
        {r.get("family") for r in dev} != {"municipal_portal", "research_lab"} or
        len({r["id"] for r in cases}) != 96):
        raise ValueError("Original H49 model-only DEV list changed")
    for row in dev:
        if (set(row) != FIELDS or row["independent_blind_test"] is not False or
            (row["width"], row["height"]) != (WIDTH, HEIGHT) or
            row["image"] != "png/" + row["id"] + ".png" or
            not re.fullmatch("[0-9a-f]{64}", str(row["image_sha256"]))):
            raise ValueError("Input includes oracle/invalid image binding")
        p = ROOT/"h49/assets"/row["image"]
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 450000:
            raise ValueError("Original bounded PNG required")
        raw = p.read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n" or sha(raw) != row["image_sha256"]:
            raise ValueError("Original public image digest mismatch")
    return dev


def adapter_gate(directory, expected):
    if not re.fullmatch("[0-9a-f]{64}", str(expected)):
        raise ValueError("Actual adapter safetensors SHA required")
    root = Path(directory).resolve(strict=True)
    if any((root / f).is_symlink() for f in
           ("adapter_model.safetensors", "adapter_config.json")):
        raise ValueError("Reject symlink adapter files")
    weights = root/"adapter_model.safetensors"
    config = root/"adapter_config.json"
    if (not weights.is_file() or not config.is_file() or
        weights.stat().st_size > 256*1024*1024 or
        config.stat().st_size > 8192 or sha(weights.read_bytes()) != expected):
        raise ValueError("Tampered or oversized model adapter")
    data = strict_json(config.read_text("utf-8"))
    if (data.get("base_model_name_or_path") != MODEL or
        data.get("peft_type") != "LORA"):
        raise ValueError("LoRA adapter base model/format differs")
    return root


def package_versions():
    import importlib.metadata
    actual = {p: importlib.metadata.version(p).split("+")[0] for p in PINNED}
    if actual != PINNED:
        raise ValueError("Unqualified pinned inference dependencies")
    return actual


def make_metadata(arm, cases, manifest, hardware, run_id, adapter_sha):
    return dict(schema=SCHEMA, arm=arm, model_id=MODEL, model_revision=REVISION,
                processor_revision=REVISION,
                prompt_sha256=digest_prompt([r["goal"] for r in cases]),
                dataset_sha256=manifest["dataset_sha256"],
                source_sha256=manifest["source_sha256"],
                capture_sha256=manifest["capture_sha256"],
                gpu_name=hardware, precision="fp16", provider="colab",
                provider_run_id=run_id, adapter_sha256=adapter_sha,
                raw_sha256="", status="incomplete")


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".temp")
    temp.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temp, path)


def run(output, arm, source_commit, run_id, adapter_dir=None, expected_adapter_sha=None):
    admit_environment()  # BEFORE importing any GPU/model package
    if arm not in ("baseline", "adapter"):
        raise ValueError("Arm must be baseline/adapter")
    if (type(run_id) is not str or not 8 <= len(run_id) <= 128 or
        not re.fullmatch("[A-Za-z0-9_.:-]+",run_id)):
        raise ValueError("Real Colab notebook session ID required")
    if (arm == "baseline" and (adapter_dir is not None or expected_adapter_sha is not None)
        or arm == "adapter" and (not adapter_dir or not expected_adapter_sha)):
        raise ValueError("Incorrect baseline/adapter checkpoint arguments")
    cases = public_cases()
    files = source_gate(source_commit)
    manifest = strict_json((ROOT/"h49/assets/sft_manifest.json").read_text("utf-8"))
    if manifest.get("independent_blind_test") is not False or manifest.get("training_executed") is not False:
        raise ValueError("Unexpected H49 public DEV source qualification")
    adapter = adapter_gate(adapter_dir, expected_adapter_sha) if arm == "adapter" else None
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Original provider output is create-only; never resubmit results")
    package_versions()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One actually allocated free Colab GPU required")
    props = torch.cuda.get_device_properties(0)
    free,total = torch.cuda.mem_get_info(0)
    if total < 10_000_000_000 or free < 8_000_000_000:
        raise RuntimeError("Insufficient free GPU memory for pinned FP16 model")
    try:
        uuid = subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],
                                       text=True,timeout=15).strip()
    except Exception as ex:
        raise RuntimeError("GPU UUID cannot be independently observed") from ex
    if not uuid or "\n" in uuid:
        raise ValueError("One physical GPU UUID must be observed")
    hardware = str(props.name) + " | " + uuid
    output.mkdir(parents=True,exist_ok=False)
    out_raw = output/(arm + ".jsonl")
    out_meta = output/(arm + ".json")
    receipt = output/(arm + "-environment.json")
    meta = make_metadata(arm,cases,manifest,hardware,run_id,expected_adapter_sha)
    _atomic_json(out_meta,meta)
    _atomic_json(receipt,{"source_commit":source_commit,"source_files":files,
        "provider_reported":"colab","provider_attested":False,
        "noncommercial_free_tier_operator_confirmed":True,
        "gpu_name":str(props.name),"gpu_uuid":uuid,
        "vram_bytes":total,"initial_free_bytes":free,
        "packages":package_versions(),"model_run_executed":False,
        "model_quality_qualified":False})
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      DO_NOT_TRACK="1",WANDB_DISABLED="true",
                      TOKENIZERS_PARALLELISM="false",HF_HOME="/tmp/hala-h49-hf-cache")
    out_raw.touch(exist_ok=False)
    try:
        from PIL import Image
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from peft import PeftModel
        processor = AutoProcessor.from_pretrained(MODEL,revision=REVISION,
                                                   trust_remote_code=False,token=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL,revision=REVISION,trust_remote_code=False,token=False,
            use_safetensors=True,dtype=torch.float16,device_map="cuda:0")
        if adapter is not None:
            model = PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
        model.eval()
        with out_raw.open("ab") as stream:
            for case in cases:
                item = dict(case_id=case["id"],language=case["language"],
                            image_sha256=case["image_sha256"],arm=arm,raw=None,
                            error="interrupted",elapsed_seconds=0.0,output_tokens=0)
                started=time.monotonic()
                try:
                    with Image.open(ROOT/"h49/assets"/case["image"]) as image:
                        if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                            raise ValueError("PNG changed")
                        rgb=image.convert("RGB")
                    messages=[{"role":"user","content":[{"type":"image"},
                             {"type":"text","text":public_prompt(case["goal"])}]}]
                    prompt=processor.apply_chat_template(messages,tokenize=False,
                                                         add_generation_prompt=True)
                    tensors=processor(text=[prompt],images=[rgb],return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        prediction=model.generate(**tensors,do_sample=False,
                                                  num_beams=1,max_new_tokens=96)
                    torch.cuda.synchronize()
                    count=int(prediction.shape[1]-tensors.input_ids.shape[1])
                    decoded=processor.batch_decode(prediction[:,tensors.input_ids.shape[1]:],
                                                   skip_special_tokens=True)[0]
                    if not isinstance(decoded,str) or len(decoded.encode())>2048:
                        raise ValueError("Model produced oversized raw response")
                    item.update(raw=decoded,error=None,output_tokens=count)
                except Exception as exc:
                    item.update(raw=None,error=type(exc).__name__[:100])
                item["elapsed_seconds"]=max(0.0,time.monotonic()-started)
                stream.write((json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
        data=out_raw.read_bytes()
        meta.update(raw_sha256=sha(data),status="completed" if len(data.splitlines())==32
                    else "incomplete")
    except BaseException as exc:
        meta.update(raw_sha256=sha(out_raw.read_bytes()),status="incomplete")
        _atomic_json(out_meta,meta)
        raise
    _atomic_json(out_meta,meta)
    evidence=str(out_raw.name)+" SHA256="+meta["raw_sha256"]
    print("HALA_H49_ORIGINAL_REMOTE_RAW",evidence,"STATUS",meta["status"],flush=True)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--arm",choices=("baseline","adapter"),required=True)
    p.add_argument("--source-commit",required=True)
    p.add_argument("--provider-session-id",required=True)
    p.add_argument("--adapter-dir")
    p.add_argument("--adapter-sha256")
    a=p.parse_args()
    run(a.output,a.arm,a.source_commit,a.provider_session_id,
        a.adapter_dir,a.adapter_sha256)


if __name__=="__main__":
    main()
