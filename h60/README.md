# H60 — Full-corpus original DragOn TRAIN ingest, private shards and real GPU training

**H60 executes the H59 plan.** The H59 handoff mapped 116 HTTP byte windows
over the four pinned original `Hcompany/DragOn` TRAIN archives but downloaded
nothing. H60 streams those exact archives, proves each complete original TAR
against the official upstream LFS SHA-256, pairs every original screenshot
with its original task JSON and publishes bounded, self-contained, provider
**private** training shards.

Base: H59 commit `1746592e726c3f94c1272e6864072808688aa679` (Draft PR #43).

## What is actually true now

| Claim | Status | Evidence |
|---|---|---|
| Original slider TRAIN TAR is byte-exact | **DONE** | whole-archive SHA-256 `2455d5b4a164d4ff3d6cf41d92d24ad2e27db29e28ea16f5585b55956356a33e` == official HF LFS digest, over all 7,146,830,848 bytes |
| Original slider images recovered | **DONE** | 100,000 original JPEGs, 0 rejected |
| Original slider drag actions recovered | **DONE** | 877,709 native actions, 0 rejected, 0 orphans |
| Upstream advertised slider count | **SUPERSEDED** | the advertised 65,000 images / 571,350 actions undercount the pinned archive; the archive itself holds 100,000 / 877,709 |
| Other three domains | **IN PROGRESS** | see `h60/MINIMAX_PROGRESS.md` |
| Real frozen Qwen3-VL-2B public-dev inference | **DONE** | `h60/real_drag_dev_baseline.py`, real T4, raw model strings preserved |
| Real gradient-bearing LoRA on the new corpus | **DONE** | `h60/native_drag_sft.py` + `training/scripts/train_adapter.py` on a free Colab T4 |
| Independent blind app-disjoint TEST | **NOT DONE** | the honest held-out source is first-party PUBLIC DEV, source-disjoint but not blind |

## Pipeline

```
h59/full_source_windows.json   116 pinned 512 MiB windows, exact byte ranges
        |
        v
h60/source_fetch.py            bounded HTTP 206 reader, per-window SHA receipts,
        |                      running whole-archive SHA over the exact bytes
        v
h60/tar_stream.py              incremental checksum-validated TAR reader; a member
        |                      that straddles a window is never split or truncated
        v
h60/pairs.py                   original image/task pairing, geometry and intent rules
        |
        v
h60/ingest.py                  one self-contained shard per bounded partition
        |                      (images + original task JSON + labels.jsonl + manifest.json)
        v
h60/publish.py                 NEW private provider dataset, authenticated is_private,
                               anonymous HTTP 403, byte-level readback SHA proof
```

### Shard layout

Each shard is one ZIP that the provider expands server-side into:

```
images/<domain>/<uuid>.jpg     original upstream JPEG bytes, untouched
tasks/<domain>/<uuid>.json     original upstream task JSON bytes, untouched
labels.jsonl                   derived training records (one per original task)
manifest.json                  per-member SHA-256, counts, source offsets, rejections
```

`labels.jsonl` rows carry the original upstream instruction, the original
`ordered` flag and the original start/end boxes in **original image pixels**;
the drag centre points are derived from those original boxes. No label, box,
dimension or task is invented.

### Rules the pipeline enforces

* Only `train/` of the pinned revision `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`.
* The public evaluation set and the hidden test set are never read.
* Every TAR header is checksum validated, so JPEG payload bytes that look like
  a header are rejected.
* The original task region is read twice: once on its own, once inside the
  full stream, and every task payload SHA must agree.
* A shard is "stored" only after the provider itself reports it, reports it
  private, rejects an unauthenticated request with HTTP 403, lists exactly the
  published member names and sizes, and returns bytes that hash to the locally
  recorded SHA-256 for **every** member.

### Commands

```bash
# one full domain, published as NEW private provider datasets
python -m h60.ingest --staging D:/hala-h60/staging --domain slider \
  --target-shard-bytes 2147483648 --result D:/hala-h60/slider-result.json

# public DEV real inference (free Colab Linux CUDA only)
python -m h60.real_drag_dev_baseline --source h55/assets --dev h55 --output <out>

# derive the trainable SFT export from verified shards
python -m h60.native_drag_sft export --source <extracted shard dir>
```

## Tests

`tests/test_h60_ingest.py` builds a synthetic original-style TAR, serves it
from a local Range-capable origin and proves the whole pipeline: boundary-safe
member recovery, image/task pairing, whole-archive SHA, window receipts and the
shard contents. It never touches the real upstream source and never touches a
provider account.
