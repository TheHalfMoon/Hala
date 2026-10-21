# Hala Training Data Library — living progress tracker
**Last verified: 2026-10-21. English code/docs/tools; updates to founder in Arabic.**

## Where everything is stored

- **CURRENT PRIVATE Kaggle library:** https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library-private ? dataset ID `12752110`. Authenticated provider listing `is_private=True`; unauthenticated Kaggle API HTTP 403. H50/H52/H53 label SHA256 remotely verified. Old previously public Kaggle slug has been deleted.
- **GitHub source registry and historical source/experiment evidence:** `research/competitor_data/2026-10-21/`, `h49/`, `h50/`, this folder and `reviews/`. Dataset/version state machine: `catalog.json`. No paid hosting or new model weights.
- **Hugging Face:** Existing OAuth connection is authenticated as `MedScale`, but has only read-oriented repository scopes (no dataset-write scope). A **new account is unnecessary**; a future write-enabled connection may simplify storing large private dataset shards if Kaggle private quotas are insufficient.
- **Storage architecture:** Kaggle private for versioned curated *small training shards*, upstream Hugging Face source repos for huge official archives, GitHub for source pins/hashes/rights evidence/tasks/replay scripts. Do not clone 200GB of AgentNet, 61GB of DragOn or all video frames onto the Windows machine. No HCompany benchmark/test trajectories included in a training archive.
- **Reproduce the first Kaggle stage:** `python -m research.data_library.build_kaggle_bundle --output NEW_NONEXISTENT_TEMP_DIR`; verify `library-manifest.json` against the original H50 source. Kaggle upload needs Kaggle API auth and is explicitly **private** (no `--public` flag). It does not run a model.

## Completion checklist — mark DONE only with real receipts

| Data source | Discovered | Source pinned & reviewed | Uploaded to Kaggle | Used in real training | Independent model eval |
|---|---|---|---|---|---|
| Original H50 bilingual train (64 unique screenshots, 384 labels) | ✅ | ✅ | ✅ **v1 private; remote SHA verified** | ☐ | ☐ |
| H52 three-step task factory (96 workflows, 288 distinct screenshots, 144 Arabic / 144 English) | ✅ | ✅ first-party source/Edge audit | ✅ **private Kaggle v2, remote SHA verified** | ☐ | ☐ |
| Original H49 DEV (32 screenshots) | ✅ | ✅ stored GitHub **not training** | — reserved DEV | — | ☐ |
| OpenCUA / AgentNet (human desktop trajectories, ~22.6k tasks) | ✅ | ☐ | ☐ | ☐ | ☐ |
| Hcompany DragOn (upstream: 286,012 train images, ~3.56m drag labels) | ? | ? pinned official slider TRAIN subset, full visual privacy audit pending | ? **87 original images / 760 native drag annotations on private Kaggle, SHA verified** | ? | ? |
| ServiceNow VideoCUA (~10k human tasks, video supervision) | ✅ | ☐ | ☐ | ☐ | ☐ |
| GUI Odyssey (~8.3k mobile trajectories) | ✅ | ☐ | ☐ | ☐ | ☐ |
| Microsoft Synthetic Computers at Scale (98 *fictional worlds*) | ✅ | ☐ | ☐ | ☐ | ☐ |
| Multimodal-Mind2Web (~7,775 train actions) | ✅ | ☐ | ☐ | ☐ | ☐ |
| WebLINX (multi-turn website interactions) | ✅ | ☐ | ☐ | ☐ | ☐ |
| GroundGUI / ScreenRef (restricted training data) | ✅ | ☐ release-scope decision | ☐ | ☐ | ☐ |
| ShowUI/Web (source imagery/rights unresolved) | ✅ | ☐ | ☐ | ☐ | ☐ |
| Holo4 evaluation rollouts (7,366) | ✅ | benchmark-only | **NOT TRAIN** | **NOT TRAIN** | reference only |
| Hcompany Agentic Task Factory | ✅ methodology | ☐ actual source access | ☐ original generated tasks | ☐ | ☐ |

**Mark a source fully DONE only after its actual source hash, train split, privacy/right review, Kaggle/HF storage verification, real model checkpoint/training-log hash and held-out evaluation receipts exist.** The checkboxes reflect exact observed stages, not the number of references found. The machine-readable `catalog.json` uses the same six stage flags and deliberately has **zero sources marked `training_completed`**.

## Evidence of Kaggle v1 upload and remote verification

Kaggle private dataset creation succeeded with Kaggle CLI 2.2.4. The Kaggle metadata API readback returned `isPrivate: true` and dataset ID `12748399`. The downloaded `library-manifest.json` was **byte-identical** to the staged copy, SHA256 `1748d1fe26cdc5218c595cf2839884a3f6b780a363b4517da69ca0c38faa67af`. The archived original upload `hala-h50-train-original.zip` SHA256 was `783426a4f8ae10c2ce85fa6b9cc861f69857e64372e1a4dc8a4cf81b8053c37c`. Kaggle unpacked the archive into a directory under `hala-h50-train-original/h50-train/`; it lists the label JSONL plus original PNG files. Download of `authoring_sft.jsonl` from Kaggle returned 240,656 bytes and original SHA256 `cb1a8529eaf0f82272d75a459c1634766ee5ec7694f6b21affa2045b11d02dd6`. **This establishes actual private remote storage for the first-party H50 TRAIN data. It does not establish training or model improvement.**

Only 64 original first-party synthetic train images, 384 source-bound counterfactual click labels, and original mock-browser provenance were uploaded. No H49 DEV, benchmark TEST answers, PHI, real government accounts, competitor images, private grants, trained weights or other user files were uploaded. Stage files under Windows %TEMP% may be cleaned **after** confirming the remote readback; do not delete the GitHub original repository artifacts.

## Data acquisition waves — why quality and action diversity matter more than raw count

1. **W1: original grounding (stored):** H49/H50 Arabic justice and traffic + English FHIR synthetic train. Checkpoint: private H50 Kaggle v1 remote verified; LoRA and model evaluation remain undone.
2. **W2: real desktop control + drag (next):** pin MIT OpenCUA AgentNet and Apache-2.0 Hcompany DragOn upstream commits and exact TRAIN partitions; inspect representative source media for identity/copyright/credentials. Add genuinely new app layouts, drag boxes and correct drag action start/end, not fake clicks. Preserve episode boundaries. Upload bounded *curated training shards* with immutable SHA and attribution to **private** Kaggle. No bulk 200GB local downloads.
3. **W3: long-horizon video/cross-app:** ServiceNow VideoCUA; GUI Odyssey true app-disjoint train; Microsoft synthetic 98 fictional computer worlds as a starting point to produce **new original task worlds**, not official evaluation answers. Align trajectory state, action, expected next observation, wrong-target recovery, and multi-step success rather than single-click rates.
4. **W4: broad web/visual language:** official `osunlp/Multimodal-Mind2Web` train only, license OpenRAIL, and WebLINX train only (CC BY-NC-SA restrictions or separately evidenced author rights). GroundGUI/ScreenRef and ShowUI remain separate pending specific source-scope/PII review; public benchmark sets are reserved for evaluation.
5. **W5: Hcompany Agentic Task Factory analogue:** derive a first-party tool that procedurally synthesizes novel bilingual task *scenes + success verifier + tool/natively executable trajectory*, with controlled difficulty, text/icon/spatial/drag/refusal/recovery strata, multiple GUIs and tools, realistic anti-shortcut evaluation and source/version seeds. Hcompany internal factory/corpus is NOT established as publicly obtainable merely because Holo4 evaluation rollouts are public. Founder's assertion of direct private permission is noted, but no sensitive documents are requested.
6. **W6: actual neural training:** start with real frozen unchanged-model grounding baseline and H49 vs H50 controlled adapters; expand to data mixtures with domain-balanced sampling, duplicate and near-duplicate filtering, confidence weights, SFT then lawful preference/RL training only on genuinely verifiable simulations and outcomes. Keep model-specific architecture (Holo4 35B-A3B final direction vs 2B pilot qualification) and GPU/free-tier limits explicit. Run separate GUI-only and GUI+tool evaluations.
7. **W7: reproducible competitiveness:** independent sealed Arabic/English novel app TEST and version-matched OSWorld 2.1, ScreenSpot-Pro/ScreenSpot-V2, drag benchmarks. Reserve those screenshots/tasks/answer trajectories from all training, self-training, teacher rewards and prompt optimization. Report successful end-to-end tasks, grounding hit, false clicks, recovery, abstain, latency, VRAM and **cost per successful task**. Never infer that more training labels alone beat competitors.

## Data sourcing notes

Primary source links and license/split details live in `research/competitor_data/2026-10-21/registry.json`, plus this folder's machine catalog. New sources verified on Hugging Face:

- https://huggingface.co/datasets/ServiceNow/VideoCUA — MIT; ~10K video tasks, 55 hours, 6M frames from 87 desktop applications.
- https://huggingface.co/datasets/microsoft/synthetic-computers-at-scale — MIT; 98 fictional workspace environments, useful for procedural scenario generation, not 98 ready-made GUI screenshots.
- https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web — OpenRAIL; official train/test splits (7,775 TRAIN actions).
- https://huggingface.co/datasets/McGill-NLP/WebLINX — CC BY-NC-SA-4.0, explicitly noncommercial/share-alike by default.
- https://huggingface.co/datasets/microsoft/Echoverse — synthetic *evaluation* environments with shipped test tasks; **not train**.

Kaggle's broad search returned weak/mismatched public results for general `computer use` and no direct AgentNet Kaggle copy, so **the original, version-pinned HF/GitHub author sources are preferable to random Kaggle mirrors**. The user stated broad Hcompany reuse permission; record it as an owner assertion and seek minimal source-level identity checks rather than requesting sensitive private legal files. Keep official TEST rollouts out of training regardless of permission, to make future accuracy claims credible.

## Honest hosted Linux CI regression and repair (2026-10-21)

Initial signed library commit `1800b6f6ddc0feaee52dfbcb896e251cccaef594` passed 3/3 local Windows tests but **FAILED** hosted Ubuntu Python 3.12 job [119065965983](https://github.com/TheHalfMoon/Hala/actions/runs/39552922307/job/119065965983); the companion Python 3.11 job was cancelled and both PR jobs initially queued. The **exact** failed test `test_builder_is_train_only_and_create_only` assumed locally rebuilt ZIP compressed bytes would be identical across operating systems; GitHub Linux `zlib` compressed the identical original entries differently (archive SHA256 `c73797bc4f00f42341eb90210c16a322db9a1b9a04473e73fa05eaac082ad788` versus original Windows archive SHA256 `783426a4f8ae10c2ce85fa6b9cc861f69857e64372e1a4dc8a4cf81b8053c37c`). That is a *test assumption error* not lost training media.

Repair verifies every original PNG and annotation **per-file SHA256**, the rebuilt ZIP's *own* calculated SHA256, and the remote-retrieved Kaggle annotation receipt separately. The immutable original Kaggle upload SHA remains preserved as its own original remote archive evidence. It does **not** change the original H50 rows, Kaggle data, source bytes, legal scope or model results. Post-fix Windows targeted 3/3 PASS, raw `reviews/hala-kaggle-library-linux-ci-repair-focused-20261021.log`; require **new exact signed-head hosted CI 3.11/3.12** success before claiming this fix qualified. Preserve historical failure without suppressing tests.

## H52 source-to-trainer admission update (2026-10-21)

- **[DONE]** First-party H52 source and 96 three-step browser replay verified; Kaggle private version 2 screenshot/annotation/manifest source bytes read back.
- **[DONE]** Bounded **source-only** H52 -> Qwen3-VL LoRA `load_authoring_records` admission through `h52/training_bridge.py` + new `h52/remote_training_manifest.json`, without changing original H52 or H50 data. Windows focused 16/16 PASS; full source 638 total, 636 PASS and 2 SKIP; Graft wiring graph clean. Receipts under `reviews/h52-training-bridge-*` and `h52/README.md`.
- **[NOT DONE]** Full Kaggle dataset single-call download/extraction: authenticated Kaggle `DownloadDataset` returned **404**; single-file original H52 PNG/JSONL downloads did pass independent SHA256 verification. No single-call full provider download qualification.
- **[NOT DONE]** Real H52 neural finetune, actual GPU allocation, trained model checkpoint, blind bilingual 3-step end-to-end task success or competitor-leading benchmark results. Source-bridge qualification is not learned-model qualification.

## H53 ? authentic DragOn shard and Kaggle privacy correction (2026-10-21)

- **[DONE]** Fetched authentic Hcompany/DragOn `train/slider.tar` on immutable commit `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5` via bounded byte ranges. Exact original UUID JPEG/JSON matching yielded 87 independent images, 760 original ordered native DRAG instructions. No eval/test sources were used.
- **[DONE]** Saved genuine training data to newly created PRIVATE Kaggle dataset ID `12752110`. Authenticated private flag true, unauthenticated HTTP 403; all three H50/H52/H53 original training JSONL source SHA256 hashes independently downloaded and matched. Removed the previous accidentally public Kaggle dataset. This does NOT prove earlier public-version access never happened.
- **[DONE]** Original H50/H52 archives retained; no third-party raw DragOn screenshots committed to the PUBLIC Hala GitHub. Kaggle license metadata for mixed-origin dataset is `apache-2.0`; original H50/H52 source remains CC0-1.0.
- **[NOT DONE]** Full original 7.1GB DragOn TAR independent checksum, image-by-image manual privacy review, LoRA/native drag model training, learned or independently benchmarked neural task success. One representative slider screenshot was visually inspected. See `h53/README.md`.

## H54 ? native ordered-drag source-to-Qwen LoRA bridge (2026-10-21)

- **[DONE]** All H53 original 87 JPEG + 760 original DragOn slider instructions replayed through source-verified JPEG-aware LoRA source loader. Genuine ordered `drag` start/end action targets are not coerced into clicks. Original 760-row derived supervised source SHA256: `205b408d314576d389fb497530fc44d115208b07d2b40501387c585c0700d40b`.
- **[DONE]** Existing click-only paths retain PNG source checks; H54 new JPEG path applies only to original H53 pinned-source data. Derived dataset is local CPU source admission, NOT a GPU training run; original source JPEG/JSON remain only in private Kaggle ID 12752110. Public GitHub stores signed source HASH receipts and source code only.
- **[NOT DONE]** Authentic source-disjoint original drag DEV baseline, approved FREE Linux CUDA, actual adapter fine-tuning, blind drag success, model weights and competitor wins. Existing H46 click baseline remains 0/8.
- **[NOT DONE]** H54 derived JSONL has NOT been uploaded as a separate private Kaggle version; it can be deterministically re-generated from original preserved H53 private Kaggle sources before a legitimate approved training run.

## H55 — real browser bilingual PUBLIC drag DEV (2026-10-21)

- **[DONE]** Created 24 original first-party fictional drag DEV screenshots on six app-disjoint scene families: 12 Arabic original public-service-inspired simulations and 12 English FHIR/design/inventory simulations. Not copied from Najiz, Absher, Hcompany, existing benchmark evaluation or real service accounts.
- **[DONE]** Genuine Microsoft Edge pointerdown/move/up replay: 24 correct ordered-drag GUI completions; 24 wrong drag endpoint rejections; 24 wrong origin rejections. Independent Python source/PNG SHA and geometry audit PASS. Original source SHA256 `707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6`. Actual original browser raw replay receipt SHA256 `cbff4d7d4db896b4a378435ffe69118c5a3d1f11e982391b934d5ad6a798d85a`.
- **[DONE]** Future raw Hala native-drag JSON evaluator: `h55/score.py` supplies geometric per-language diagnostic; `h55/replay_predictions.js` *actually executes* unchanged supplied raw model action proposals in Edge and records accepted/rejected GUI task success. Harness itself has no neural model and no paid services.
- **[DONE: HARNESS UNIT TEST ONLY]** Original source-golden mock responses, explicitly flagged `synthetic_unit_fixture`, replayed in genuine Edge **24/24** accepted; deliberately wrong end coordinates replayed **0/24** accepted. These are positive/negative **tester unit inputs**, NEVER model inference or learned task success. Original unit evidence is in `h55/evidence/`.
- **[NOT DONE]** Genuine frozen Qwen unchanged-model raw drag predictions, adapter training, GPU optimizer steps, real neural end-to-end drag task success, actual blind independently sequestered holdout or competitor benchmarking. H55 is **PUBLIC DEV, not a BLIND TEST or training data**. Do not import H55 labels/images into the private Kaggle TRAIN dataset.
- **[DONE]** Windows full source: 659 total tests, 657 PASS, two existing SKIP, zero failures, plus Graft wiring build/check PASS (253 files, 2052 nodes, 5476 edges). Exact new signed HEAD GitHub CI is NOT qualified until all hosted jobs are terminal. See `h55/README.md` and `reviews/h55-public-dev-*`.
