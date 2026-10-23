# H70 — Complete Original DragOn TRAIN CPU Domains and Private Recovery

**Status:** Three remaining H61 original TRAIN domains exported and packaged
on the CPU, full Kaggle private provider revalidation pending independent
authenticated byte readback for each. No GPU, no weights trained, no release.

Parent PR #53 (H69 original slider verified private storage). Branch
`ps/h70-original-three-domains-private-qualification`.

## Actual CPU exports

On the user's authorized Windows WSL Ubuntu environment (Pillow 11.3.0,
Python 3.11, H61 code from pinned Git), all three exact original
source TRAIN domains were exported with HTTP-206 range integrity
receipts. The source image and native action origins remain pinned to
upstream revision `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`.

| Domain | Source TRAIN images | Native drag actions | Private archive bytes | Archive SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `sheet` | 2,500 | 64,954 | 633,422,412 | `c8bef672f6fc491c994e6ddf322a5d2d371162531c169965b746ce99da24f2e9` |
| `slide_resize` | 2,500 | 26,313 | 438,021,742 | `be80ebe6a48dcf6d7d39317992926eeae428afd16e949e7e71fed9e203ba7c82` |
| `text_highlight` | 2,500 | 24,990 | 347,938,610 | `0bb0520b7714db7193af671aaed621f9c931376637f1661de26dbeedb09ba721` |

All three archives were built using `h69.domain_archive pack`, which
validates exact domain counts, original native action associations, JPEG
byte digests, original TRAIN source revision and byte-range receipt
counts. Each archive is immutable and source-Git bound to
`3250c6062575848fef379e5cda8bc3cfb1611166`.

**Do not equate CPU packing with Kaggle durability** until Kaggle returns
each complete private binary, anonymous listing is denied, SHA256 matches,
and the source-bound H69 unpack and revalidation passes.

## H70 private provider protocol

`python -m h70.publish_original` performs a fail-closed upload and
verification of a specific archived original domain, rejecting a wrong
digest before touching the provider. It uses the owner's existing Kaggle
CLI authentication without printing tokens or using a paid service.

After actual owner-private publication it requires a complete provider
listing, anonymous HTTP 403, an independently downloaded, SHA-matched
archive, and source-level rederivation after readback. A successful
result is `PRIVATE_BYTE_READBACK_VERIFIED`, not model training.

Kaggle dataset creation is eventually consistent: an accepted private
create can precede a visible `datasets files` listing. H70 retries that
read-only check and supports `--resume-existing` to complete an
already-created dataset **without attempting a second publication**.

Negative tests explicitly refuse wrong archive SHA, public endpoint,
missing/corrupt provider bytes, duplicate receipt or accidental reuse of
the H69 `slider` dataset. CI must be green on Python 3.11 and 3.12.

## Scientific boundaries

- Original pinned source CPU dataset now includes **11,000 images**
  across four DragOn TRAIN domains including H69's 3,500 slider images.
  Total H69+H70 original native drag annotations: **147,097**.
- Separately, H68 created **2,000 original browser-rendered
  bilingual TRAIN scenes**, 1,000 per language, and restored the
  Kaggle-private full binary with independent byte verification.
- The final H61 curriculum of **10,000 selected TRAIN rows** has not yet
  been assembled or stored privately.
- This phase contains no optimizer step, new H61 checkpoint,
  real GUI accuracy improvement, H63 paired-image model evidence, or
  general computer-use release. The most recent measured H60 GUI
  success remains 0/48; keep no-change claims factual.
- No source screenshots, Kaggle tokens or weight files may be committed
  to public GitHub. Keep the work as a stacked Draft PR.
