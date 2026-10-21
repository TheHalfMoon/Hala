# H59 — Hala Million-Example Training Map for MiniMax

**Master handoff dated 2026-10-21. Owner: TheHalfMoon/Hala.** This is a living execution map. It distinguishes upstream availability, training-file materialization, provider-verified private storage, executed model training, and independent benchmark outcomes. Do **not** conflate any two states.

## What is actually ready right now

| Dataset | Upstream available | Locally source-verified | Independently verified PRIVATE Kaggle | Actual neural training |
|---|---:|---:|---:|---:|
| H Company DragOn slider (H53) | 65,000 screenshots / 571,350 tasks | 87 JPEG / 760 drag tasks | 87 / 760 in [Hala GUI Training Library PRIVATE](https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library-private) | None |
| H Company DragOn sheet, slide resize, text highlighting (H57) | 221,012 screenshots / 2,988,792 tasks | 117 JPEG / 1,586 drag tasks | 117 / 1,586 in [Hala DragOn Multidomain TRAIN PRIVATE](https://www.kaggle.com/datasets/abdulazizshehri/hala-dragon-multidomain-train-private) | None |
| DragOn combined (four TRAIN domains) | **286,012 images / 3,560,142 TRAIN instructions** | **204 original images / 2,346 instructions** | **204 / 2,346 verified original samples** | **0 gradient steps** |
| H50/H52 first-party | 64 H50 images, 384 targets; 288 H52 screenshots, 96 episodes | Recorded in source catalog | Existing Hala GUI Training Library PRIVATE | None |
| AgentNet (22.6K trajectories), VideoCUA (~10K tasks / 6M VIDEO FRAMES), GUI Odyssey (~8,334 mobile episodes) | Published upstream | Source/metadata researched | **Not bulk imported or claimed ready** | None |
| H55/H58 first-party original Edge | 24 slider +24 mixed-domain PUBLIC DEV screens | Genuine author-oracle browser tests | **Never TRAIN. Not in private TRAIN datasets** | None |

**Upstream DragOn source pin:** `Hcompany/DragOn@3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`, source license Apache-2.0, original `train/` TAR objects only. The official full-source train TAR lengths were independently checked at the pinned Hugging Face dataset tree on 2026-10-21:

| Source TAR | Original images | TRAIN annotations | TAR bytes |
|---|---:|---:|---:|
| `train/text_highlight.tar` | 100,000 | 1,000,000 | 29,874,024,960 |
| `train/sheet.tar` | 38,332 | 996,632 | 14,258,661,376 |
| `train/slide_resize.tar` | 82,680 | 992,160 | 9,915,637,248 |
| `train/slider.tar` | 65,000 | 571,350 | 7,146,830,848 |
| **Full upstream TRAIN** | **286,012** | **3,560,142** | **61,195,154,432 (~61.2 GB)** |

Only a very small authentic subset of these full TARs was actually materialized before H59. **A remotely accessible source is not 3.56 million Kaggle-ready rows.** No complete TAR SHA has yet been independently verified and no full original source training has occurred.

## Immediately usable full-source fetch map

`h59/full_source_windows.json` SHA256 **`a2e99696970d83aa9e50a12a659ea7f3c79def6841333cedb7c5a9091de44b5b`** is a complete **116-window 512-MiB-or-less HTTP Range inventory** covering all **61,195,154,432** original source bytes, without gaps or overlapping planned ranges, pinned original revision, domain, start/end byte, source URL, expected exact 206 Content-Range, **and all four actual upstream Hugging Face LFS SHA-256 file digests**. The source SHA values are upstream-declared and become independently verified only after `h59/verify_full_source.py` reconstructs all byte windows and validates each entire original TAR digest. **All 116 windows initially have `window_downloaded=false`, `source_window_sha256=null`, `reusable_train_examples_ready=0`.** No SHA should ever be invented. This ~90 KB plan is intended to be mirrored to a new **PRIVATE** Kaggle metadata dataset that MiniMax can attach to its data worker.

The read-only source downloader `h59/fetch_window.py` downloads **one** selected window at a time when explicitly invoked with `--download`, requires the correct pinned SHA URL, enforces actual HTTP206 + Content-Range and exact byte count, streams in ≤4MiB blocks, atomically records a full window SHA256 and receipt, and supports resuming an incomplete `.part` only within that same fixed range. A downloaded TAR byte window **is not a TAR member or a usable image/task example**. Whole-member-aware downstream extraction must handle original TAR headers and members that span multiple byte windows (including pax/longname headers if present), and must not mistake an incomplete JPEG/JSON for a complete example.

```bash
python -m h59.master_data_plan verify --file h59/full_source_windows.json
# On an authorized CPU/data preparation runtime only; 512 MiB free working capacity or more:
python -m h59.fetch_window --plan h59/full_source_windows.json \
  --window-id h59-sheet-0000 --output-dir /kaggle/working/hala-source-shards --download
# Raw staged windows are not 'private stored' until a NEW private Kaggle dataset
# has been created and its full SHA256 independently read back.
```

**Source retention and privacy:** Never push external H Company JPEG/task originals, source TAR bytes, model weights, Kaggle credentials, or any private screenshot into PUBLIC GitHub or Vercel workflow logs. Preserve license/attribution notices accurately. Do not use `eval/`, official benchmark trajectories or privately sealed TEST for training. Follow Kaggle's private dataset quota and auto-unpacking behavior; make each uploaded data shard a **NEW PRIVATE dataset**, verify authenticated `is_private=True` and anonymous `403` before considering a shard stored. Do not accidentally create/update a public dataset or reuse deleted former public Hala slug.

## Platform / long-running operation reality

**Kaggle:** Its current official notebook technical page specifies **up to 12 hours per CPU/GPU run**, only **20 GB autosaved `/kaggle/working`**, other scratch may disappear, and GPU quota is **limited (typically about 30 hours/week; verify actual account balance)**. Kaggle's dataset page lists **200GB per dataset, 200GB total private datasets**, and up to 50 top-level files, with ZIPs unpacked as virtual files. 61.2GB upstream TAR bytes nominally fit under 200GB only if this particular account has sufficient remaining quota and extracted dataset expansion does not exceed it; **account headroom has not been measured**. Never assume 12-hour sessions or 30-hour weekly quota can be circumvented or extended by restarting jobs.

**Usage eligibility gate:** The currently inspected Kaggle ToS language is described as personal, internal, **non-commercial** usage; Hala's intended future integration in Aladdin is commercial. The user's H Company source-code/data permission does **not** automatically grant permission from Kaggle to train a commercial product. MiniMax must first check the current binding Kaggle terms or obtain explicit Kaggle permission suitable for the intended training. While this remains unresolved, **metadata/data preparation and noncommercial research planning may continue only within permitted usage; do not disable the existing `train_adapter.py` Kaggle-training restriction or launch commercial GPU training**. If ineligible, transfer the identical plan to another expressly authorized no-cost compute provider (such as permissible Colab/free university allocation) without changing source/test boundaries.

**Vercel Workflow:** Workflow SDK gives durable state, `use workflow` / `use step`, sleep/retry/hooks or webhooks, **not GPU compute**. Its Hobby plan has free but finite limits; no Vercel workflow app has yet been deployed or had live credentials verified. If MiniMax already runs a separately permitted Vercel Workflow installation, it can poll Kaggle job status and maintain resumable **external orchestration**. Every Kaggle session still ends when Kaggle ends it. Do NOT use persistent loops, tunnels, multi-account quota evasion, idle keepalive, proxy GPUs or forced restarts. Treat `QUEUED` as queued, never as trained. Do not resubmit existing H47 queued notebook.

**Required durable job states:** `discovered → pinned → window_downloaded → window_sha_verified → original_image_task_indexed → deduplicated → split_verified → provider_private_uploaded → provider_private_readback_sha_verified → curriculum_ready → GPU_eligibility_approved → real_baseline_done → actual_optimizer_steps_done → raw_neural_eval_done → independently_blind_eval_done`. Status transitions require immutable receipts; never set a later status based only on a plan file. Each retry idempotently checks original unique job key = `{source_commit, domain, byte-window-id, source_sha256, transform_revision}` and does not repeat completed verified work.

## MiniMax execution phases and gates

| Gate | Goal and work | Required proof before DONE |
|---|---|---|
| G00 | Inspect exact GitHub H58/H57 frontier, permissions, provider account limits, Kaggle usage eligibility, Vercel pricing/quota | Current GH SHA, owner account identity, budget cap $0, provider eligibility decision |
| G01 | Use existing private H50/H52/H53/H57 samples now; audit all already stored original sample hashes, keep originals untouched | Private dataset ID, expected train-only artifact SHA and provider readback |
| G02 | Materialize 116 pinned source windows, **CPU preparation first**, never GPU for copying files. Create resumable provider-private chunks, preserving source offset/bytes/sha | 116 range receipts/validated 61.2GB source without gaps; actual TAR-level SHA if feasible |
| G03 | Parse all original TAR members across window boundaries, reconstruct original JPEG/JSON UUID pairs, derive source-correct `ordered` native drag start/end boxes with original pixels; exclude incomplete/tampered pairs | For every source record: image SHA, task SHA, train split, original UUID/offset, image dimensions, action semantics, rejection log |
| G04 | Deduplicate and leakage control by image SHA, source family, app/page template, contiguous trajectory; create independent held-out app/domain datasets **without training on official benchmark eval/test** | Whole-train record counts vs official 286,012 images / 3,560,142 tasks; exact unique counts; audited split IDs and exclusion reasons |
| G05 | Stage only completed, verified, bounded shards in **PRIVATE** Kaggle within real storage headroom, record original and derived SHA for each; use HF remote source if quota prevents replication | Full provider-private readback for each shard; no secret/public leaks |
| G06 | Make separate curricula: drag ordered/unordered; click/type/scroll; multi-step actions; Arabic/English originals and verified instruction augmentation. Create first-party verifier Task Factory, avoiding copied benchmark answers | Per-skill counts, firstparty synthetic provenance, dynamic mixture, image/task ratio, failure examples, train/test separation |
| G07 | **Before any optimizer step:** on authorized no-cost CUDA, freeze pinned unchanged Qwen/Qwen3-VL-2B base; run real raw neural inference on original H55/H58 PUBLIC DEV and a separately sequestered BLIND TEST; score actual Edge task completion | Real unmodified model output strings, exact model git SHA, GPU memory, runtime, 24+ original scene results, true deficits |
| G08 | If provider use/VRAM/budget eligible, start real gradient-bearing LoRA on H54/H57 small source first; then staged 10k→100k→1m→3.56m samples. Checkpoint with optimizer/scheduler/scaler/RNG/adapter and source SHA; fail closed if no finite gradient update | Step loss/gradient norms, optimizer updates, checkpoint SHA, resume identical-state checks, training source consumption counts |
| G09 | Add tool sequence / recovery, multilingual, abstain/refusal, and a verifiable sandbox reward only after source-qualified SFT. RL/DPO only on authentic preferences/rewards, no reward hacking | Real postcondition success on separate apps, error taxonomy, pause/stop safety, Arabic and English per-skill results |
| G10 | Compare unchanged base, each adapter and eligible competitors on identical released versions, identical benchmark harness/splits, held-out tasks and locked compute; no unverified SOTA claims | Raw results, matched benchmark tables + CIs/uncertainties, time and cost, test contamination audit |
| G11 | Release provenance/licensing/privacy/security/reproducibility/packaging; only mark complete when all real hardware and scientific gates pass | Actual model checkpoint, license eligibility, hardware inference test, source manifest, versioned eval receipts |

**Training time feasibility:** 3,560,142 annotations do not fit in a free 12h GPU session with a nontrivial 2B multimodal model. Organize by **verified token/image budget**, not nominal task count. Check actual images/VRAM/OOM and run controlled LoRA ranks/adapters before considering the larger Holo4-35B-A3B target. Single free T4/P100 cannot be assumed to support 35B model training. Train only in provider-authorized sessions and resume legitimately within quotas.

## Model competency and quality criteria

Required skill coverage: click, type, scroll, drag, resize, text highlight, selection, keyboard shortcuts, menus, multi-app workflows, GUI state validation, recovery from failed actions, no hallucinated actions and robust refusal on unsafe tasks. English/Arabic native screenshot support (translating English instructions **does not create an Arabic UI screenshot**). AgentNet desktop human trajectories, VideoCUA videos and GUI Odyssey mobile episodes are **separate skills/corpora**. VideoCUA's ~6 million frames are **frames, not 6 million independent labeled action tasks**. Curate permissions/splits/format and prove source fetch before adding each as training ready.

Metric **actual successful browser/environment task completion**, weighted across actions, languages and apps, with latency/memory and safe action refusal. For original DragOn `ordered=false` tasks, reversed valid drag endpoints are equivalent; for `ordered=true`, reversals are errors. Do not report 100% synthetic author-oracle harness as neural accuracy. Benchmark training contamination, independent TEST custody, probabilistic uncertainty and repeated trials are mandatory before a 'best computer-use model' claim.

## Durable evidence and cleanup policy

- Save authoritative **non-sensitive** plans, step receipts, catalog, test logs, source hashes, real evaluation raw model outputs and every negative outcome to GitHub. Do not store H Company original restricted raw media or private model weights publicly.
- Save original actual TRAIN, real training shards, adapters and resume state only in a confirmed PRIVATE artifact store or authorized model repository with sufficient quota. Kaggle `/kaggle/working` only autosaves at completed versions; checkpoint **well before** the 12h deadline (e.g. at 6h/10h if time limits allow) and independently verify remote artifacts before deleting scratch.
- Never restart an already queued Kaggle H47 run or replace existing source shards without an immutable-source-digest and generation-ID check. Cancel/pause when authorization, privacy, quota, or GPU goes out of bounds. Minimax should continue independently only on work it actually controls.
- After each stage, mark **DONE / FAILED / BLOCKED / RUNNING** in the repository's living source catalog with an actual link and original SHA. A Stage marked 'planned' is NOT finished.

### Source documents

- [Official H Company DragOn dataset](https://huggingface.co/datasets/Hcompany/DragOn)
- [Kaggle Notebooks technical specifications](https://www.kaggle.com/docs/notebooks)
- [Kaggle private dataset technical specifications](https://www.kaggle.com/docs/datasets)
- [Kaggle GPU weekly limits](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle CLI kernel push/status/output](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Vercel Workflow SDK](https://vercel.com/blog/a-new-programming-model-for-durable-execution)
- [Vercel Workflow model](https://github.com/vercel/workflow)
- [Vercel Hobby limits](https://vercel.com/docs/plans/hobby)

## Durable source-preparation state contract (implemented, not deployed)

`h59/durable_contract.py` provides a deterministic idempotency key and strict state admission per original source byte-window ID. States are `PLANNED → WINDOW_SHA_VERIFIED → ORIGINAL_MEMBERS_VERIFIED → PRIVATE_KAGGLE_READBACK_VERIFIED`. Source commit/byte ranges are immutable; state regressions and skipping proof gates are rejected. Provider-private advancement also requires a matching original-member SHA, an original image/task member count, authenticated private-status evidence and unauthenticated 403 evidence. It never permits any source-preparation job to claim GPU optimizer steps or model success. For any later states, MiniMax must independently query actual provider metadata; this is a **pure CPU schema validator**, not a Vercel deployment, provider check or proof that the real files exist.

```bash
python -m h59.durable_contract --plan h59/full_source_windows.json
python -m unittest tests.test_h59_durable_contract -q
```

**23 focused offline tests (source map + Kaggle private handoff + durable contract)** passed as of 2026-10-21. Real 116-window downloads, all 3.56m records, Vercel orchestration deployment, and GPU model training remain **NOT DONE**. Vercel Workflow is optional coordinator: it should call this state admission contract from a genuine bounded step and persist state, but cannot create free Kaggle GPU time, overcome 12-hour sessions or authorize a prohibited usage. No Vercel credential or deployment was accessed by this work.
