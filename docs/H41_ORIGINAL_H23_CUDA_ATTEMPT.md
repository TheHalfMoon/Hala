# H41 — retrospective execution of untouched historical H23 CUDA lifecycle

**Status:** source-only private GPU launcher prepared; no H41 real CUDA result until a new Kaggle Batch attempt yields and an independent verifier validates raw records. This is **research-only**, no paid services, no production deployment and no host OS clicks.

## Why an untouched original H23 commit

H23's SHA-frozen first-party source was historically reviewed at **`f104e3e68fe378faf65c63248cb239bd2d171c28`**. The original freeze SHA256 is **`4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a`** and binds **53 original Python/config files**. A clean detached checkout at the old commit exactly passed `evaluation.serving_lifecycle.verify_sources`.

The current H40/H41 branches include four additional first-party Python files and changes to `hala/inference/vision_candidate.py`, so simply running the old H23 `--freeze-sha256` at current HEAD would fail the source gate. Do **not** rewrite/refreeze the old source to erase that evidence requirement. H41 instead checks out its authentic historical SHA from GitHub, proves clean source and the original freeze, then executes the original runner unchanged with its exact parameters.

## Private Kaggle Batch experiment and hard gates

`tools/build_h41_h23_kaggle.py` generates one **create-only** private Kaggle notebook `abdulazizshehri/hala-h41-original-h23-cuda-lifecycle-20261020`. It must only run under permitted free Kaggle GPU quota, with explicit research opt-in. It performs the following in strict order:

1. Observe actual Linux Kaggle Batch marker, noncommercial research opt-in and a CUDA device without importing model weights. Reject local host, paid alternatives, ambiguous origin and existing output paths.
2. Clone the H41 source branch, fetch and checkout the **original H23 SHA**, require `git status --porcelain --untracked-files=all` empty. Reject modified source and freeze.
3. Verify the original 53-file first-party freeze. Align **only** ephemeral Kaggle Pillow to historically required `11.3.0` (if necessary) using exact package pin and `--no-deps` with bounded timeout, then require actual `torch 2.11.0`, `transformers 5.16.1`, Pillow `11.3.0`, RAQM and installed DejaVuSans SHA256 `ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280`. No permissive fallback if any pin fails.
4. Run **unchanged** `evaluation.serving_lifecycle --provider kaggle --freeze-sha256 4c522... --source-commit f104... --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf --output /kaggle/working/hala-h41-h23-run --faults` as an owned process with an explicit 1000-second parent timeout, exact process-group termination/reap on timeout. Both historical scripts and original H23 source remain immutable.
5. Retain raw output/results/wire trace/progress, operator log and preflight/runtime metadata. Always keep model/native/CUDA cancellation **unqualified** in launch metadata. The source itself also records `actual_cuda_cancel_qualified=false` until independent judgment.

## Not a commercial or native action test

The H23 'GUI' tasks are two **first-party synthetic settings workflows**, including one Arabic fixture; they do not interact with real external apps. The operator uses its own ephemeral HTTP loopback and owned worker process. No host/device/UI clicks are authorized.

The actual H23 CUDA cancellation criterion requires a genuinely pending CUDA matrix operation from the loaded model with recorded enqueue time **before** cancellation timestamp and still pending at the worker's acknowledgment, timestamp provenance `event_set`, and exact worker reap/restart evidence. A toy CUDA tensor, CPU-fixture acknowledgment, unrelated allocation or a successful Kaggle status does **not** qualify this criterion.

**Important caveat:** Unlike H22's same-allocation precision comparison, this new private Kaggle Batch attempt is on a *new GPU allocation*. It cannot retrospectively prove same-allocation H22 comparability and must be reported separately with observed hardware identity. If Kaggle environment doesn't match, retain fail-closed error and do not change source safeguards. A CI-green notebook generator is not a CUDA observation.

## Local and source gates

```powershell
python -m unittest tests.test_h41_h23_frozen_gpu_notebook -v
python -m unittest discover -s tests -q
python -m compileall -q tools/build_h41_h23_kaggle.py
```

Run notebook builder only from an exact signed committed H41 source; do not push hidden API credentials or model weights to Git. The source-bundle generator and the preserved historical source are separate from the private Kaggle GPU worker and later independent evidence audit.

**Remaining:** authentic GPU run and independent model-event audit, source-independent security reviews, real-app screenshots, independently controlled English/Arabic evaluation and founder production decision. Never label this as Hala finished.

## Actual Windows source qualification before remote submission

- **H41 focused source-freeze/notebook tests: 3/3 PASS**, including live historic `git show` hash verification of all 53 original source files, private-only notebook config, non-opted-in hard rejection, create-only output and syntactic validation of both notebook cells.
- Full Windows serial Python 3.14.7 suite **536 total / 534 passed / 2 skipped**, **104.875 seconds**, exit 0. Log `reviews/h41-source-suite-20261020.log` SHA256 `073644e9ed4a864a1e0174885424fb77df3e54c69eaab3c0dc51d8b4793b8d62`; credential-pattern scan zero matches. Compileall PASS.
- Graft wiring-only build/check PASS **1,655 nodes / 4,476 edges** across 193 parsed files; deep semantic tier **not** built.
- Free Kaggle user quota checked: **28.94 GPU-hours remaining** at last inspection, but this is not proof of successful CUDA allocation. Actual provider telemetry and source/weights must be captured separately.
- Original H23 source SHA `f104e3e68fe378faf65c63248cb239bd2d171c28` remains unchanged. H41 source-level CI, real provider run, log/result replay and independent reviews are distinct pending gates.
