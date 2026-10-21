# H60 / MiniMax — live execution ledger for the full DragOn TRAIN corpus

**Started 2026-10-21 on branch `ps/h60-million-drag-train-ingest`, stacked on the
H59 handoff `1746592e726c3f94c1272e6864072808688aa679` (Draft PR #43).**
All counts below are measured, not advertised. "DONE" always carries a receipt.

## Platform eligibility actually established on 2026-10-21

| Provider | Measured fact | Decision |
|---|---|---|
| Kaggle **GPU compute** | Current Terms of Use §3: *"You will only use the Services for your own internal, personal, non-commercial use, and not on behalf of or for the benefit of any third party."* Hala is intended for a future commercial Aladdin product. | **BLOCKED.** The `train_adapter.py` and `baseline.py` Kaggle guards are **not** disabled. No Kaggle GPU session was launched. |
| Kaggle **private dataset storage** | Authenticated owner `abdulazizshehri`; three pre-existing PRIVATE datasets; measured download **32.6 MB/s**, measured upload **~1.0 MB/s** single stream and **~3.3 MB/s** with four concurrent uploads. | **USED** for private source shards only, consistent with the founder's own authorized H53/H57/H59 lineage. No public dataset, no Kaggle compute, no credential in any log. |
| Google Colab free T4 | Official `google-colab-cli` 0.7.4 under WSL, authenticated as `alshehriofficial@gmail.com`; allocated `gpu-t4-s-kkb-ass1b1-2zhirjoi8lkx7`, **Tesla T4, 15,637,086,208 bytes**, torch `2.11.0+cu130`; `colab usage` reports **balance 0.00 compute units** before and after allocation, so no paid compute unit was consumed. | **USED** for the real inference and the real LoRA run. Google's terms permit organisational research use; no paid runtime, no subscription, no overage. |
| Vercel Workflow | No Vercel account, CLI, credential or deployment is present in this environment. | **NOT DONE.** Orchestration stays portable in `h59/durable_contract.py`; no running service is claimed. |

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| Pin the original source and the four TRAIN archives | **DONE** | `Hcompany/DragOn@3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`, Apache-2.0, `train/` only |
| Recover the **complete original slider TAR** byte-exactly | **DONE** | all 7,146,830,848 bytes streamed; SHA-256 `2455d5b4a164d4ff3d6cf41d92d24ad2e27db29e28ea16f5585b55956356a33e` equals the official upstream LFS digest |
| Recover the original slider images and drag actions | **DONE** | 100,000 original JPEGs, 877,709 native actions, 0 rejected rows, 0 orphan tasks, 0 images without a task |
| **Correct the upstream advertised slider count** | **DONE** | the advertised 65,000 images / 571,350 actions undercount the pinned archive; the archive itself holds **100,000 / 877,709** |
| Recover the other three original TRAIN archives | **IN PROGRESS** | `sheet`, `slide_resize`, `text_highlight`; see `h60/MINIMAX_PROGRESS.md` below |
| Publish private provider shards with byte-level readback | **IN PROGRESS** | one NEW private dataset per bounded shard; authenticated `is_private`, anonymous HTTP 403, exact provider listing and per-member SHA readback |
| **Real frozen Qwen3-VL-2B inference on first-party PUBLIC DEV** | **DONE** | 24/24 H55 + 24/24 H58 cases on a real Tesla T4 at revision `89644892e4d85e24eaac8bacfd4f463576704203`; raw model strings preserved verbatim |
| Measured baseline GUI drag task success | **DONE (negative)** | **0/24** geometrically correct on H55 and **0/24** on H58; strict bare-JSON validity 0/24 and 0/24 |
| **Real gradient-bearing LoRA on authentic private DragOn TRAIN** | **RUNNING** | `training/scripts/train_adapter.py`, FP16 rank-8 LoRA on the unchanged pinned model, free Colab T4 |
| Recoverable checkpoint and resume | **PENDING** | must save optimizer/scaler/RNG state and prove a resumed run matches |
| Genuinely private app-disjoint **blind** TEST | **NOT DONE** | the honest held-out source is first-party PUBLIC DEV: source-disjoint, but not blind and not app-disjoint from a third party |
| Official benchmark comparison | **NOT DONE** | no benchmark eval/test source was ever read |

## Measured baseline detail (real, unchanged model, no adapter)

| Set | Cases | Strict bare-JSON valid | Tolerant parse | Ordered semantics valid | Geometrically correct | Degenerate start==end | Out-of-range points |
|---|---:|---:|---:|---:|---:|---:|---:|
| H55 bilingual drag PUBLIC DEV | 24 | 0 | 24 | 24 | **0** | 21 | 0 |
| H58 mixed-domain PUBLIC DEV | 24 | 0 | 16 | 16 | **0** | 6 | 8 |

The frozen 2B model emits markdown-fenced JSON instead of the requested bare
object, frequently returns an identical start and end point, and never lands a
drag endpoint. That is the measured deficit the LoRA curriculum must attack.

## Provider publication path: measured corrections (2026-10-21, later in the same session)

The first publication attempt failed for reasons that were **not** source
problems. Each correction below is a measured provider behaviour, and each
correction now has a test.

| Measured provider behaviour | Consequence for the pipeline | Correction |
|---|---|---|
| `kaggle datasets files` serves **20 rows by default** (max 200) and the provider API paginates the dataset index | the readback compared only the first page against a 28,000-member inventory and every large shard was reported as a failed readback even though the upload had succeeded | the provider index is now read with explicit pagination, and the authoritative readback is a single streamed provider download whose **every member is hashed locally** |
| a dataset created seconds earlier is **refused with HTTP 403** until the provider serves the new version | the propagation delay was recorded as a privacy failure | the pipeline waits for the provider itself to serve the dataset before any privacy or readback proof is attempted |
| the provider CLI host (`api.kaggle.com`) is throttled with **HTTP 429** while `www.kaggle.com/api/v1` stays available | readback was unreliable and slow | the readback uses the provider REST endpoints; every CLI call retries with backoff on 429 and 5xx |
| the provider may return the expanded tree **wrapped in one common directory** | a byte-exact member comparison would have rejected a correct dataset | the readback maps provider members onto published members and still requires an exact name, size and SHA-256 match for every member |

Two further properties of the publication path were changed so that a
half-published shard can never be mistaken for a stored shard:

* the local shard body is released **only after** the provider has been proven
  to hold exactly the published bytes, so an interrupted publication is
  retried from the same local body instead of re-downloading the source;
* a failed readback **deletes** the half-published provider dataset and
  records a named failure receipt;
* every shard member carries a fixed timestamp, so an identical shard body
  always produces identical archive bytes and the shard SHA-256 becomes a
  stable resume key: an already verified shard is recognised and is **not
  uploaded twice**.

The one half-published H60 slider dataset of the superseded generation was
deleted rather than left in place; the three pre-existing founder datasets
were not touched.

### Publication-path receipts

* **DONE** Provider publication probe with synthetic non-original bytes:
  dataset created, `provider_is_private=true`, anonymous view **HTTP 403**,
  readback of **20/20 members** with **0** mismatched bytes, provider index
  complete, probe dataset deleted afterwards.
* **DONE** `tests/test_h60_publish.py`: **8/8 PASS** — deterministic shard
  bytes, release-only-after-verified-readback, failed-readback cleanup,
  resume without a second upload, provider index pagination, wrapped-tree
  readback, tampered-member rejection, missing-member rejection.
* **DONE** `tests/test_h60_ingest.py`: **8/8 PASS** — synthetic original-style
  archive over a local Range origin: boundary-safe member recovery,
  image/task pairing, whole-archive SHA, window receipts, shard contents. The
  suite never touches the real upstream source and never touches a provider
  account.

## Honest limits of this ledger

* The private provider copy of the corpus is a **staging copy of a corpus that
  is already publicly and permissively available at the pinned upstream
  source**. The upstream source is the authoritative copy; the private copy is
  the provider-verified mirror with per-member SHA evidence.
* No Kaggle GPU compute was used, so no commercial-use restriction was
  circumvented. No Kaggle quota was consumed.
* The PUBLIC DEV sets are **not** training data and were never used to derive
  a training label.
* Nothing here claims a competitive or SOTA result, and no independent blind
  TEST has been run.
* The provider upload bandwidth measured at roughly 1.0 MB/s on one stream and
  3.3 MB/s on four concurrent streams is the binding constraint on how fast
  the full 61.2 GB corpus can be mirrored privately. It is a transfer
  constraint only; it does not limit the corpus size.
