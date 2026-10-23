# H66 — Exact screenshot cap, measured against original DragOn TRAIN

**Branch:** `ps/h66-exact-source-cpu-stage` based on H65 `1c1386acec3893c8ebb6c464eb2c5ddc1652269a`.
**GPU use:** none. **H61 trained model outcome:** still unmeasured.

## Real source observation (2026-10-23)

The unchanged H61 exporter operates on pinned DragOn TRAIN byte ranges, but the
stop condition checked `len(wanted) >= images_wanted` **after** feeding a
complete HTTP chunk into the TAR stream. A single 4 MiB chunk included 68
accepted JPEG members, so a requested five images actually exported 68.
This made the bounded-request setting inaccurate.

The new code refuses further matching image members inside `keep_image`
and independently checks the limit inside `sink_image`. Existing original
source offsets, byte-range integrity checks, screenshot transformations,
original action annotation extraction, provenance and official split exclusion
remain unchanged. The fix only affects future exports and does not rewrite
or relabel previously qualified Stage C data.

### Two actual WSL CPU exports on the same source

Both ran via `python -m h61.colab_run --steps export --images slider=5
--chunk-bytes 4194304`, with a temporary isolated Pillow 11.3.0 environment;
not a Colab GPU. Source original upstream revision remains pinned in H61.

| Metric | Original H64 source | Patched H66 source |
| --- | ---: | ---: |
| `images_wanted` | 5 | 5 |
| `original_images` | **68** | **5** |
| `native_drag_actions` | 598 | 47 |
| `image_region_bytes_read` | 4,194,304 | 4,194,304 |
| `task_region_bytes_read` | 915,404,288 | 905,969,664 |
| `range_receipts` | 29 | 28 |
| `elapsed_seconds` | 78.478 | 77.300 |
| Receipt SHA-256 | `587b18a8fda28cf9c8b7062ad502d5d71a05adade73b0bf7ca62b166b12cf39e` | `60cf92d68afbe54382f64934bbcdc2a638386668e58d0ccfd49937e3d7c7d8ea` |

The corrected run is **more precise about the requested dataset cardinality**,
but the nearly identical elapsed time is negative evidence for any substantial
speed improvement. The source task region still requires reading about 906MB
for a five-image training pilot, because original task members are stored
separately from the JPEG members. Do not promise that batching alone solves
CPU/GPU preparation costs.

## Repeatable deterministic test

`tests/test_h66_exact_source_cap.py` reproduces 68 matching TAR members in
one fake HTTP range, proves that requesting 5 retains exactly 5, requesting 1
retains 1, and requesting 68 retains 68. **3/3 PASS** on an independent
isolated WSL checkout.

No fake screenshot or original source image was pushed to GitHub. The pilot
raw downloads and temporary isolated interpreter are staging, not durable
training data; only the counts and receipt digests are preserved here.

## Next real training bottleneck

**Do not spend another short-lived T4 on unchanged `export,firstparty,curriculum`**.
The H61 CPU-only steps must run first, then be packaged with source/byte
integrity identities in a private durable TRAIN staging dataset. Only after
authenticated full readback should a free GPU session perform the actual
weight training and matched evaluation. H65 checkpoint egress stays required
during optimizer steps.

No blind benchmark, geometric grounding gain, or released model exists from H66.
