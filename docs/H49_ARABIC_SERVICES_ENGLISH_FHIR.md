# H49 — Original Arabic public-services + English FHIR visual grounding

**Date: 2026-10-21. Research-only, noncommercial, original synthetic offline GUI dataset. NOT a real Najiz/Absher/clinical user journey, learned model result or production authorization.**

Source branch: stacked on H48 first-party Chromium SFT [PR #36](https://github.com/TheHalfMoon/Hala/pull/36), current parent signed `54baccb55681ea12e5d08eed4809d095c2fee4fd`. Preserves all historical H14/H23/H41/H46/H47/H48 original source, frozen manifest, negative model scores and dedicated queued H47 Kaggle notebook. Uses only authorized Windows device Abdulaziz, no Mac, paid API or local model weights.

## Domain selection: separate language policy and real-world caveats

| Synthetic original family | Language | Split | Inspiration / observable skill |
|---|---|---|---|
| `justice_execution` | Arabic RTL only | TRAIN | Public Najiz service categories: request lookup, filing, evidence, scheduling, notifications; original paraphrased fictional labels |
| `traffic_services` | Arabic RTL only | TRAIN | Public Absher traffic and vehicle service categories: insurance, impound, appointment, driver license, citations, registration |
| `fhir_resources` | English only | TRAIN | HL7 FHIR **R4** resource selection: `Observation`, `DiagnosticReport`, `Patient`, `Encounter`, `MedicationRequest`, `Practitioner` |
| `fhir_bundles` | English only | TRAIN | HL7 FHIR **R4** `Bundle.type`: `searchset`, `transaction`, `document`, `message`, `collection`, `history` |
| `municipal_portal` | Arabic RTL only | PUBLIC DEV | Original fictional municipal permit application, distinct app/template/family |
| `research_lab` | English only | PUBLIC DEV | Original fictional non-patient research-lab workflow, distinct app/template/family |

**Official source references — vocabulary and semantics only (no scraped site HTML, logo, screenshot or proprietary asset):**

- Najiz official public services: https://najiz.sa/applications/landing/
- Absher official public portal and service guide: https://www.absher.sa/
- HL7 FHIR R4 Observation: https://hl7.org/fhir/R4/observation.html
- HL7 FHIR R4 Bundle: https://hl7.org/fhir/R4/bundle.html

All training labels, page markup, tasks, paraphrases, panel styles and screenshots are newly authored first-party CC0-1.0 fictional mock content. The presence of real product/service names in these **research notes** is descriptive only. Screens deliberately say they are independent research simulations, not official services. No government ID, plate, user record, medical patient, clinical data, credential, portal login, OTP, payment, case submission, CAPTCHA or session automation is collected or accessed. No real public website was opened by the capture script: only local `file:` pages were allowed, with every other request aborted. There is no authority to operate on real accounts.

## Delivered experiment assets and independent verifiers

`h49/factory.py` deterministically authors **96 offline HTML cases** across six fictional app families, 16 examples per family: 64 TRAIN (32 Arabic, 32 English) and 32 PUBLIC DEV (16 Arabic, 16 English). Each task presents six plausible service/resource options, shuffled across target positions, and an independently paraphrased semantic goal. No privileged target index/DOM box is included in a model-facing prompt.

`tools/h49_offline_capture.js` captures original 1024×640 Edge Chromium PNGs using pinned local `puppeteer-core 25.12.0`, verifies all six controls are fully visible within the unobscured viewport, then performs **one real browser DOM click on the intended option** and on a different distractor after resetting the page. These are **local mock-browser DOM clicks**, not actual real application or native OS actions. The resulting `h49/assets/capture.json` records every raw image digest/size, six measured DOM rectangles, correct selected key and wrong-key rejection. It does not make network requests to Najiz, Absher, HL7 or other domains.

`h49/audit.py` independently reconstructs the original source, checks original HTML **byte-for-byte**, PNG signatures/dimensions/SHA256, browser control text, labels and geometry, 96 golden and 96 wrong-option postconditions, full target click uniqueness, language/domain family isolation and explicit no-real-service claims. It exports create-only:
- `h49/assets/model_inputs.json`: 96 model-facing public synthetic input descriptions **without** target index/key/DOM truth
- `h49/assets/evaluator_only.json`: 96 labelled evaluation oracle cases, stored separately; **public and not blind**
- `h49/assets/authoring_sft.jsonl`: 64 train-only window-pixel click completions
- `h49/assets/sft_manifest.json`: source/capture/JSONL/media digests, rights, language/family and nonqualification metadata
- `h49/assets/integrity.json`: counts of the actual browser replays and source checks

Training input ingestion **reuses** the existing `training/scripts/train_adapter.py` with an explicit H49 schema and an H49-only byte-for-byte provenance-bound source loader. The H48 route remains supported and the H14-frozen `training/artifacts.py` is entirely unchanged. No unqualified model weights or tokenizers are downloaded.

Repro from repository root (only in a clean new output path, never overwrite original evidence):

```powershell
python -m h49.factory --root PATH_TO_EMPTY_NEW_OUTPUT_ROOT
$env:PUPPETEER_CORE_PATH = 'PATH_TO_INSTALLED_PINNED_PUPPETEER_CORE_25_12_0'
node tools/h49_offline_capture.js PATH_TO_NEW_OUTPUT_ROOT
python -m h49.audit --root PATH_TO_NEW_OUTPUT_ROOT
python -m unittest tests.test_h49_domain_grounding -v
python -m unittest discover -s tests -q
```

The original pinned output `h49/assets` is create-only. Keep original evidence unchanged, compare new independent runs by digest and report source/runtime/browser version changes. Never silently replace receipts or upgrade pinned captures.

### Genuine negative during build

The very first Edge capture attempt **failed** on `h49-traffic_services-01`, because the sixth button was partially offscreen and the attempted click did not register a DOM postcondition. The failure and independent root-cause replay are kept in `reviews/h49-original-capture-first-failure.md`. Repair compressed the original sidebar row heights, added explicit **fully onscreen DOM geometry admission**, and regenerated this new H49 source only before successful capture. The original H48/H47/H14 assets were neither touched nor refrozen.

### Scientific limits / next steps

These are original simulated single-click tasks, not a license to operate Najiz/Absher accounts and not a real OSWorld 2.1/ScreenSpot-Pro score. TRAIN and public DEV families are separated, but shared authorship, public oracle Git files, similar component grammar, publicly known goals and screenshot layouts mean **no independently blind heldout or generalization claim**. Arabic human review of transliteration, legal service terminology, and RTL accessibility remains outstanding.

**No actual H49 model training has occurred.** Before LoRA/adapted-model comparisons: verify the intended baseline SHA/tokenizer/processor and provider license, free compute terms and quotas, pre-register a *new independently sequestered* family/app-disjoint bilingual TEST set and score identical checkpoints under the same hardware/harness. The existing remote trainer still blocks Kaggle for eligibility and requires an exact real failed baseline; this work does not relax either condition. The original H46 0/8 outcome is retained honestly, H47 queued kernel must not be resubmitted, no commercial release or competitor-win claim is authorized.

CI, full serial local tests, genuine Graft graph integrity and optional independent D1/PStack/Alibaba semantic reviews must be stated from **actual** logs at the final signed SHA, never assumed from a draft or earlier head.

## Genuine engineering qualification — 2026-10-21

At initial **signed and GitHub Verified** H49 code head `73b404173bdc749b72cb714e45ce337f9fa2e1fb` (parent `54baccb55681ea12e5d08eed4809d095c2fee4fd`):

- Original headless Edge offline capture: **96/96 original PNGs**, **576 browser-measured controls**, **96/96 golden next-option browser DOM state observations**, **96/96 wrong-option postconditions rejected by the source verifier as not satisfying the goal**. This is *not* 96 model predictions, protected navigation or live service execution. Browser raw `h49/assets/capture.json` SHA256 `2bd300775bfa739dd7f88b2e1cbc2539fa9f47bdc015930b596460460b23fbf4`; canonical source SHA256 `b9498312f459a6f3f54891e9e8736a0042f4805ec561abc787c890b7f07019fb`.
- Original `h49/assets/authoring_sft.jsonl`: SHA256 `df242de75f52b0401c1767fc94bbc2e24e4d10a17b0a7174da0e599c89bfc8cd`, **64 fully original bilingual annotations**. These are correct *authoring labels*, not measured model accuracy.
- Focused Windows CPU tests: **9/9 PASS** including immutable H14 source, historical H48 loader support, full original 96 screenshot verifier, strict Arabic vs English family policy, coherent SFT forgery and hard negative controls. Raw `reviews/h49-domain-focused-20261021.log` SHA256 `ed626c543bf1a7976439e6c275e90e2c7784921a0f5ef0fd7dfb6c69fa1296b3`.
- **Full Windows serial suite: 587 tests, 585 PASS + 2 skipped, 0 failure/error**, exit code 0; `reviews/h49-full-suite-20261021.log` SHA256 `4f7bda732552065324850542435cb2d39bcf86792345ac9a6bf4963de3aa7446`. The two skips are the suite's existing environment-dependent skips, not suppressed newly failing tests.
- Genuine local Graft **wiring graph build + source-check PASS**, **1,841 nodes, 4,975 edges across 225 files**. `reviews/h49-graft-build-20261021.log` SHA256 `7e18432a16baca5e01a7cd33899f2d7af97c037fbb069524217e5011f3235e5c`; `reviews/h49-graft-check-20261021.log` SHA256 `cd28f8506c9d483217c1d5ffff51c53a9f98ab2b5b0b07c6e32a925bd6d37916`. Graft semantic/deep meaning tier was **NOT RUN**.
- Initial exact source head push/PR GitHub Actions run [39527319159](https://github.com/TheHalfMoon/Hala/actions/runs/39527319159) and [39527425085](https://github.com/TheHalfMoon/Hala/actions/runs/39527425085): all four hosted Python **3.11 and 3.12 jobs SUCCESS**. Any later evidence-only commit must be qualified against its own exact new head before readiness.
- Explicitly **NOT PERFORMED:** paid or Kaggle training, any real Najiz/Absher credentials/account interactions, FHIR patient data/real EHR workflow, medical/pharmaceutical action, proprietary screenshot capture, private independent blind TEST, deep Graft semantic audit, independent D1/PStack/Alibaba semantic approval, native OS E2E or original-vs-adapted model performance comparison. The previous genuine H46 **0/8 original model** result remains unchanged, and the queued H47 Kaggle notebook is not resubmitted.

Keep this PR **DRAFT and unmerged** until required policy/independent review and model-validation gates are legitimately satisfied. No claims of Arabic government-app mastery, English FHIR workflow autonomy, OSWorld success, or better-than-Holo4 performance arise from these synthetic assets.

## Paired real-inference scoring contract (post-data engineering)

The CPU-only `h49/paired_eval.py` is the next engineering gate toward actual **learned grounding**: it checks two *externally produced* fully recorded real-model DEV inference arms, original model versus independently verified trained LoRA adapter on the exact same original 32 PUBLIC DEV screenshot+goal pairs. This change **does not train, download or execute any model** and must never publish a score using the unit tests' explicitly fake input data. Live GPU execution remains unavailable/unqualified in the current Windows checkout.

Before executing a real experiment: source-pin original `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203` (model card advertises Apache-2.0 at https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct, but recheck exact revision and all transitive licenses). Verify model/processor match, actual *free* approved Linux Colab GPU resource and terms, actual nonzero adapter gradient updates and immutable signed adapter checkpoint and hash. Colab free GPU access is variable/not guaranteed per https://research.google.com/colaboratory/faq.html. No paid Google AI, commercial inference or Kaggle training restriction bypass.

### Exact cross-arm I/O schema

Use the existing `training.scripts.train_adapter.prepare_inputs` image-plus-goal chat-template construction style for inference. The exact prompt function `h49.paired_eval.public_prompt(goal)` reproduces every frozen H49 SFT training prompt, so there is no silent incompatibility between the SFT output schema and the inference format. Both arms see **only** the 1024×640 PNG and that public user prompt, never `h49/assets/evaluator_only.json`, `capture.json`, source expected target, indexed DOM controls or privileged label. Do not expose the private TEST set to either inference arm. Public DEV is for diagnostics only.

Each arm must produce an independent unchanged raw `*.jsonl` file: exactly 32 rows, **in the order of the existing DEV rows in `h49/assets/model_inputs.json`**, with exactly the following keys:

- `case_id`, `language`, `image_sha256`, `arm` (`baseline` or `adapter`).
- `raw`: the original unedited decoded model output string (at most 2048 UTF-8 bytes) or `null` on error; `error`: `null` on success or a bounded type/name on failure. Never postprocess/correct the model's raw JSON.
- `elapsed_seconds`: a finite nonnegative actual measured inference duration; `output_tokens`: a nonnegative integer observed from the GPU decoder.

For each arm also save a `*.json` metadata file, with **exact** fields: `schema` (`hala.h49.paired-raw-inference.v1`), `arm`, `model_id`, `model_revision`, `processor_revision`, `prompt_sha256`, `dataset_sha256`, `source_sha256`, `capture_sha256`, `gpu_name`, `precision` (`fp16`), `provider` (`colab` or an authorized `linux_cuda` provider), `provider_run_id`, `adapter_sha256` (null for baseline, exact 64-digit hex for real adapter), `raw_sha256`, `status` (`completed`). All source and prompt hashes must be regenerated from the frozen manifest; do not invent a GPU name or run identifier. Save actual provider notebook logs and attested adapter/model SHA separately for independent qualification.

The scorer explicitly requires externally supplied SHA256 digests for **both** metadata and raw outputs for each arm; only compares 32/32 complete cases with identical model/processor/hardware/precision/goal/prompt/source and individually verified screenshot digests. It recomputes source images, all original DOM rectangles and distinct accepted targets, then reports by language and all: parse invalid, error, abstain, wrong unique control, unbound/ambiguous control, correct target, latency and tokens, plus paired baseline-only/adapted-only wins, both wins and both wrong. It refuses changed, reordered, duplicate, missing or internally inconsistent evidence. It records `real_model_provenance_independently_verified=false` **even if metadata looks authentic**, because a checked file hash is not proof of actual GPU provider execution.

Example only (fill paths and all four *actual external* digests from genuine runs):

```bash
python -m h49.paired_eval --root h49/assets \
  --baseline-meta real-run/baseline.json --baseline-raw real-run/baseline.jsonl \
  --baseline-meta-sha VERIFIED_REAL_SHA --baseline-raw-sha VERIFIED_REAL_SHA \
  --adapter-meta real-run/adapter.json --adapter-raw real-run/adapter.jsonl \
  --adapter-meta-sha VERIFIED_REAL_SHA --adapter-raw-sha VERIFIED_REAL_SHA \
  --output real-run/independent-public-dev-score.json
```

**Before a real model inference producer is approved or any competitive assertion**: require actual provider original signed notebook source and immutable artifacts, independent GPU/run evidence, honest failures, a frozen baseline on same prompts, and a *separately created, sealed, app-family-disjoint and bilingual* TEST set. H49's DEV Arabic `municipal_portal` and English `research_lab` are not the Arabic Najiz/Absher or English FHIR TRAIN families, so this DEV scoring measures cross-domain transfer, **not** heldout same-service mastery. No numbers from synthetic unittest fixtures constitute empirical model accuracy.

The scorer can be exercised on local synthetic fake data via `python -m unittest tests.test_h49_paired_eval -v`; those fake examples have clearly marked fictitious provider IDs and **must never** be archived or summarized as neural performance.

### Exact-score-harness engineering evidence — 2026-10-21

The source-bearing scored-raw-evidence extension is signed and GitHub Verified at commit `9395e6ec6bf0ebce930999b078af137843a8b676` on the same Draft PR #37 (no duplicate PR). Exact source-head GitHub Actions runs [39530449013](https://github.com/TheHalfMoon/Hala/actions/runs/39530449013) (push) and [39530452086](https://github.com/TheHalfMoon/Hala/actions/runs/39530452086) (PR) have **four SUCCESSFUL jobs**, Python 3.11 and 3.12 on both runs.

- Actual Windows offline paired scorer focused suite: **8/8 PASS**, wholly fabricated memory-only parser and negative-path fixtures; `reviews/h49-paired-scorer-focused-20261021.log` retained (no generated inference outputs or false quality estimates).
- Actual Windows complete serial Hala suite after introducing the scorer: **595 tests, 593 PASS and two existing skipped, zero failure/error**, exit 0, 343.132 seconds reported by unittest. Raw `reviews/h49-paired-scorer-full-suite-20261021.log` SHA256 `207b7040c987a2db8aca74558060f25524c4d7d31be904cea20ea59f7767a1b5`. The slow Windows run was not aborted or represented as a performance qualification.
- Genuine Graft **wiring-only** build/check: **PASS 1,868 nodes, 5,046 edges, 227 files**. Raw `reviews/h49-paired-scorer-graft-build-20261021.log` and `reviews/h49-paired-scorer-graft-check-20261021.log` retained; semantic/deep meaning analysis **NOT RUN**.
- Neither the scoreboard script nor its tests produced real neural outputs. A real Colab GPU allocation, authorized model+adapter fit, externally attested raw baseline/adapted inferences, a separately sealed independent bilingual TEST and genuine version-matched competition remain **NOT PERFORMED**. Local independent D1/PStack/Alibaba code review was unavailable, so no approval is claimed.

An evidence-only later commit does not change the scored protocol, but its own exact-head hosted CI must be checked separately before considering this PR's last commit qualified. PR remains **DRAFT, unmerged**.

## H49 genuine remote model inference producer — staged only, not executed (2026-10-21)

`h49/remote_infer.py` closes the concrete missing producer between existing H49 32-case public DEV inputs and `h49/paired_eval.py`. It includes the exact original frozen Qwen3-VL-2B Instruct checkpoint revision, original first-party PNG digests, identical SFT/DEV instruction, bounded generation `do_sample=False`, and direct optional `PeftModel.from_pretrained` for a *genuinely trained* adapter from `training/scripts/train_adapter.py`. It does not load an oracle, original DOM/target labels or benchmarking TEST answers. It has no Windows/Mac model-loading route and does not run automatically as CI.

### Hard experiment admission (before large weights)

The runner first rejects anything except an **actual free noncommercial Google Colab Linux runtime**, with `COLAB_RELEASE_TAG`, `HALA_H49_FREE_COLAB_RESEARCH=1` and `HALA_H49_NO_PAID_RUNTIME_CONFIRMED=1`. Those two opt-ins express a human operator's **genuine confirmation of account billing plan and exact free GPU session**: the code cannot independently verify free account billing, or promise a GPU is available. Do **not** set either for a paid/pro runtime. Refuse Kaggle, other cloud runtimes, local Windows or Mac.

It checks clean content of exact published `--source-commit` via `git show` of every public DEV PNG and the input manifest plus both inference protocol scripts (same SHA as remote checkout), strict 32 actual source-bound cases, one real CUDA GPU with at least 10GB total/8GB free, hardware UUID from `nvidia-smi`, and pinned versions of torch 2.11.0, Transformers 5.16.1, PEFT 0.20.0 and Pillow 12.3.0. The package pins are inherited from Hala's prior real H15/H34 path and must be installed on the remote kernel with compatible free-tier CUDA drivers. Availability **NOT VERIFIED HERE**.

A single **real free Colab session** must use the same nonempty `--provider-session-id` for both arms, and the same physically observed GPU UUID in resulting `gpu_name`. Do not fabricate an ID or claim it proves Google attestation. The adapter arm additionally requires the actual local PEFT `adapter_model.safetensors` and `adapter_config.json`, verified SHA256 and matching pinned Qwen base model. Source must be a genuine completed adapter checkpoint with signed training/update evidence, not a synthetic file or a checkpoint copied from unqualified training. The script verifies file bytes but cannot independently prove checkpoint training.

**Important:** A new real **unchanged-model baseline** must be measured and admitted **before** any new H49 LoRA training. The existing training preflight already demands real deficient unchanged-model evidence and rejects Kaggle; neither rule is bypassed. The H49 remote producer is for collecting a baseline and, later, separate adapter inference on unchanged public DEV; it does **not** itself execute training, source fine-tuning or submit an online notebook.

Colab execution only after the operator confirms an actually allocated free GPU, source signed SHA and pinned packages:

```bash
# One genuine free Colab runtime, with an authorized clean published Hala checkout.
# NEVER run these commands on Windows/Mac or paid Colab.
export HALA_H49_FREE_COLAB_RESEARCH=1
export HALA_H49_NO_PAID_RUNTIME_CONFIRMED=1
SOURCE_COMMIT=ACTUAL_SIGNED_40_CHAR_H49_SHA
SESSION_ID=ACTUAL_COLAB_SESSION_ID_USED_IN_BOTH_ARMS
python -m h49.remote_infer --arm baseline --source-commit "$SOURCE_COMMIT" \
  --provider-session-id "$SESSION_ID" --output /content/h49-baseline-evidence
# After a verified actual adapter is trained (and only in the same free GPU session):
python -m h49.remote_infer --arm adapter --source-commit "$SOURCE_COMMIT" \
  --provider-session-id "$SESSION_ID" --adapter-dir /content/VERIFIED_TRAINED_ADAPTER \
  --adapter-sha256 ACTUAL_TRAINED_ADAPTER_SAFETENSORS_SHA256 \
  --output /content/h49-adapter-evidence
```

Each attempt is persisted as an unedited `baseline.jsonl` or `adapter.jsonl` with raw decoder output, actual time, output tokens, error type, image and case binding. Each arm gets scorer-compatible `baseline.json`/`adapter.json`, an additional full source/runtime/UUID receipt, and SHA256s. Partial/provider-failed runs remain `incomplete`; no scoring or experiment success is permitted from them. Preserve original artifacts and provider logs in signed GitHub before deleting temporary Colab output, and only then run the independent scorer with four exact raw/metadata SHA256 anchors. No live portal login, protected action, medical data, paid API, weights on the personal computer or benchmark TEST answers.

### Current outcome/limits

**No remote Colab session was available/provisioned or launched in this round.** There is no new model inference, adapter training, GPU measurement or measured accuracy improvement from this producer. Its CPU-only Windows unit tests verify *admission, source binding and output-contract construction* using first-party H49 public mock images and explicitly fake adapter bytes, but they are not model metrics. Repository docs are in English; communicate findings to the founder in Arabic. Keep H49 Draft/unmerged pending actual legal, GPU, checkpoint and independent benchmark qualification.
