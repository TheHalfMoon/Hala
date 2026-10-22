"""H61 the real remote run: export, curriculum, train, evaluate, publish receipts.

This is the durable orchestration entry point. One command performs the whole
measured sequence on an authorised free Colab Linux CUDA session:

1. `export`      read the pinned original DragOn TRAIN archives with HTTP 206
                 byte ranges and a per-range SHA-256 receipt, and materialise a
                 bounded original screenshot set per domain;
2. `curriculum`  build the capability-balanced, screenshot-deduplicated
                 curriculum from those originals;
3. `train`       run real gradient-bearing LoRA training with the
                 coordinate-focused supervision and an immutable checkpoint
                 bundle at every step;
4. `evaluate`    run the matched public DEV protocol with the trained adapter
                 through the same runner and the same scorers that measured
                 the frozen base;
5. `report`      write the receipts that bind the whole run together.

Every step is idempotent: a finished step is detected and skipped, an
interrupted training run resumes from its own published checkpoint, and a
step that cannot complete leaves the previous receipts in place.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "hala.h61.colab-run.v1"
FIRSTPARTY_SCENES = 2000
DOMAIN_BYTES = {
    "slider": 7146830848,
    "slide_resize": 9915637248,
    "sheet": 14258661376,
    "text_highlight": 29874024960,
}


def canonical(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(payload))
    return path


def step_export(work: Path, domains: dict, chunk_bytes: int) -> dict:
    """Materialise the bounded original corpus for every requested domain."""
    from h61.source_export import export_domain

    receipts = []
    for domain, images in domains.items():
        target = work / "export" / domain
        receipt_path = target / "receipt.json"
        if receipt_path.exists():
            receipts.append(json.loads(receipt_path.read_text("utf8")))
            print(f"H61_STEP export {domain} already complete", flush=True)
            continue
        total = DOMAIN_BYTES[domain]
        print(f"H61_STEP export {domain} images={images}", flush=True)
        receipt = export_domain(
            domain,
            total,
            target,
            images_wanted=images,
            chunk_bytes=chunk_bytes,
            receipt_dir=target / "range-receipts",
        )
        receipts.append(receipt)
        print(
            f"H61_STEP export {domain} done images={receipt['original_images']} "
            f"actions={receipt['native_drag_actions']}",
            flush=True,
        )
    return {"domains": receipts}


def step_curriculum(work: Path, target_rows: int, max_rows_per_image: int,
                    family_targets: dict | None = None) -> dict:
    """Build the capability-balanced curriculum from the exported originals."""
    from h61.curriculum import export as export_curriculum

    source = work / "export"
    rows_path = source / "h61_drag_curriculum.jsonl"
    manifest_path = source / "h61_drag_curriculum_manifest.json"
    if rows_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf8"))
        print("H61_STEP curriculum already complete", flush=True)
        return {
            "curriculum_rows": manifest["train_rows"],
            "unique_source_images": manifest["unique_source_images"],
            "capability_counts": manifest["capability_counts"],
            "language_counts": manifest["language_counts"],
            "selected_by_family": manifest.get("selected_by_family"),
            "dataset_sha256": manifest["dataset_sha256"],
        }
    result = export_curriculum(
        source, target_rows=target_rows, max_rows_per_image=max_rows_per_image,
        family_targets=family_targets,
    )
    print(f"H61_STEP curriculum done rows={result['curriculum_rows']}", flush=True)
    return result


def step_train(work: Path, config_path: Path, max_steps: int, resume: bool) -> dict:
    """Run the real gradient-bearing LoRA training with a verified checkpoint."""
    from training.scripts.train_adapter import train

    source = work / "export"
    output = work / "training"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text("utf8"))
    config["max_steps"] = max_steps
    effective_config = work / "training-config.json"
    write_json(effective_config, config)
    baseline = work / "public-dev-baseline-receipt.json"
    result = train(
        str(effective_config),
        str(source / "h61_drag_curriculum.jsonl"),
        str(source / "h61_drag_curriculum_manifest.json"),
        str(baseline),
        str(output),
        resume=resume,
    )
    print(f"H61_STEP train done steps={result.get('step')}", flush=True)
    return result


def step_evaluate(work: Path, repo: Path, dev_sets: list[str]) -> dict:
    """Run the matched public DEV protocol with the trained adapter."""
    from h60.real_drag_dev_adapter_eval import run as run_adapter_dev

    results = {}
    for dev in dev_sets:
        output = work / "eval" / f"adapter-{dev}"
        if (output / "run-result.json").exists():
            results[dev] = json.loads((output / "run-result.json").read_text("utf8"))
            print(f"H61_STEP evaluate {dev} already complete", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        source = repo / ("h55" if dev == "h55" else "h58") / "assets"
        print(f"H61_STEP evaluate {dev}", flush=True)
        results[dev] = run_adapter_dev(
            str(source), str(output), str(work / "training"), dev=dev
        )
    return results


def step_official_eval(work: Path, repo: Path, limit: int, domains: list[str], label: str) -> dict:
    """Measure the frozen base or the trained adapter on the official split."""
    from h61.official_eval_run import run as run_official

    target = work / "official-eval" / label
    if (target / "official-eval-run.json").exists():
        print(f"H61_STEP official-eval {label} already complete", flush=True)
        return json.loads((target / "official-eval-run.json").read_text("utf8"))["domains"]
    adapter = None
    adapter_digest = None
    if label == "adapter":
        from h60.real_drag_dev_adapter_eval import verify_adapter_checkpoint

        identity = verify_adapter_checkpoint(str(work / "training"))
        adapter = identity["adapter_dir"]
        adapter_digest = identity["adapter_weights_sha256"]
    print(f"H61_STEP official-eval {label}", flush=True)
    receipt = run_official(domains, str(target), limit=limit, adapter=adapter,
                           adapter_sha256=adapter_digest)
    return receipt["domains"]


def step_firstparty(work: Path, repo: Path) -> dict:
    """Build the first-party bilingual drag corpus on the remote session.

    The corpus is authored by the committed generator and the committed pages,
    and the pixels are rendered by the genuine browser of the session that
    trains on them. Every rendered screenshot is audited pixel by pixel
    against the recorded handle and target boxes, so a scene that did not
    render where it was authored is rejected with a reason.
    """
    from h61.firstparty_drag import build as build_firstparty

    target = work / "export" / "firstparty"
    if (target / "receipt.json").exists():
        receipt = json.loads((target / "receipt.json").read_text("utf8"))
        print("H61_STEP firstparty already complete", flush=True)
        return receipt
    receipt = build_firstparty(target, FIRSTPARTY_SCENES)
    print(
        f"H61_STEP firstparty done scenes={receipt.get('scenes_accepted')} "
        f"languages={receipt.get('by_language')}",
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--steps", default="export,curriculum,baseline,official_base,train,evaluate,official_adapter")
    parser.add_argument("--images", default="slider=3500,slide_resize=2500,sheet=2500,text_highlight=2500")
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--max-rows-per-image", type=int, default=1)
    parser.add_argument("--family-targets", default=None,
                        help="JSON object of family to maximum row count")
    parser.add_argument("--family-targets-file", default=None,
                        help="file holding the JSON object of family to maximum row count")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dev-sets", default="h55,h58")
    parser.add_argument("--official-limit", type=int, default=30)
    parser.add_argument("--official-domains", default="slider,sheet,slide_resize,text_highlight")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["HALA_REMOTE_TRAINING_ALLOWED"] = "1"
    os.environ["HALA_REMOTE_BASELINE_ALLOWED"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    started = time.time()
    domains = {}
    for item in args.images.split(","):
        name, _, count = item.partition("=")
        domains[name] = int(count)
    family_targets = None
    if args.family_targets_file:
        family_targets = json.loads(
            Path(args.family_targets_file).read_text("utf-8-sig")
        )
    elif args.family_targets:
        family_targets = json.loads(args.family_targets)
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    receipt = {
        "schema": SCHEMA,
        "repo": str(repo),
        "work": str(work),
        "steps": steps,
        "domains": domains,
        "target_rows": args.target_rows,
        "max_rows_per_image": args.max_rows_per_image,
        "family_targets": family_targets,
        "max_steps": args.max_steps,
        "resume": args.resume,
        "provider": "interactive_approved_free_colab",
        "paid_compute_units_consumed": 0,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "results": {},
        "failures": [],
    }
    write_json(work / "run-receipt.json", receipt)
    try:
        for step in steps:
            if step == "export":
                receipt["results"]["export"] = step_export(work, domains, args.chunk_bytes)
            elif step == "curriculum":
                receipt["results"]["curriculum"] = step_curriculum(
                    work, args.target_rows, args.max_rows_per_image, family_targets
                )
            elif step == "firstparty":
                receipt["results"]["firstparty"] = step_firstparty(work, repo)
            elif step == "baseline":
                from h61.baseline import build_baseline_receipt

                baseline = build_baseline_receipt(
                    repo / "h60" / "evidence" / "public-dev-baseline-20261021.json"
                )
                write_json(work / "public-dev-baseline-receipt.json", baseline)
                receipt["results"]["baseline"] = baseline
            elif step == "train":
                receipt["results"]["train"] = step_train(
                    work,
                    repo / "experiments" / "runtime" / "h61-colab-train-config.json",
                    args.max_steps,
                    args.resume,
                )
            elif step == "evaluate":
                receipt["results"]["evaluate"] = step_evaluate(
                    work, repo, [item for item in args.dev_sets.split(",") if item]
                )
            elif step == "official_base":
                receipt["results"]["official_base"] = step_official_eval(
                    work, repo, args.official_limit,
                    [item for item in args.official_domains.split(",") if item],
                    "frozen-base",
                )
            elif step == "official_adapter":
                receipt["results"]["official_adapter"] = step_official_eval(
                    work, repo, args.official_limit,
                    [item for item in args.official_domains.split(",") if item],
                    "adapter",
                )
            elif step == "report":
                pass
            else:
                raise ValueError(f"Unknown H61 step: {step}")
            write_json(work / "run-receipt.json", receipt)
    except BaseException as error:  # noqa: BLE001
        receipt["failures"].append(f"{type(error).__name__}: {error}"[:400])
        receipt["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(work / "run-receipt.json", receipt)
        print(f"H61_RUN_FAILED {type(error).__name__}: {error}", flush=True)
        raise
    receipt["elapsed_seconds"] = round(time.time() - started, 3)
    receipt["status"] = "completed"
    write_json(work / "run-receipt.json", receipt)
    print("H61_RUN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
