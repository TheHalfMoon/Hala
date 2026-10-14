# H14 source and remote evaluation

CPU implementation is experimental. No model result or native qualification is
implied by the synthetic oracle/negative controls. The immutable manifest and its
externally recorded SHA-256 establish a prospective source/task/prompt freeze.
Never re-freeze this version or tune prompts after observing heldout inference;
repairs after evaluation require a new declared experiment and prospective tasks.

Before inference, author/freeze explicitly from the repository root:

```text
python -m tools.freeze_heldout
python -m unittest discover -s tests -p test_heldout.py -v
python -m unittest discover -s tests -v
```

Existing pre-inference freezes require `--replace-pre-inference` and a fresh
recorded hash. Review/qualification evidence remains separate and parent-owned.
The freeze captures source byte hashes, including this runner and protocol, not
just a dataset count. Source line ending changes invalidate it.

On an authorized free remote Linux CUDA allocation only, inspect and install the
exact versions in `heldout_runner.VERSIONS`: torch 2.11.0 (CUDA wheel suffix is
recorded), transformers 5.16.1, Pillow 11.3.0. Use an actual RAQM-enabled Pillow
build and declare the DejaVu Sans font file. The font and shaping version are
recorded; Arabic fails closed if RAQM is absent. No local model downloads.
Reuse the previously inspected remote `/tmp/hala-model-cache` through `HF_HOME`
so the unchanged pinned checkpoint can use its existing remote cache.

```bash
export HALA_REMOTE_HELDOUT_ALLOWED=1
python -m evaluation.heldout_runner --manifest-sha256 RECORDED_SHA256 --output /tmp/hala-h14-run --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

The output directory must be new. First `image-freeze.json` persists all 64 PNG
and RGB hashes, font bytes, image dimensions/DPI and shaping provenance. Only then
does the single shared pinned unchanged Qwen3-VL-2B checkpoint load. Every image
is rechecked before its inference call. Raw model input text is only goal and
protocol; screenshots contain visible UI content without expected-state markers.

`result.json` persists after each call and on failures. A blocked or partial run
has no full-run aggregate. It records seed, deterministic algorithms, FP16/SDPA,
bounded greedy generation (48 tokens, 15-second generation stopping criterion),
30-minute experiment alarm, first cold call versus later warm calls,
preprocessing/inference/end-to-end time, process CPU and wall seconds, and allocator
allocated/reserved/peak bytes plus CUDA device free/used/total bytes. Device used
memory includes other processes and driver allocations; allocator figures do not.
Generation max_time is checked between decoding steps; the outer alarm bounds the
experiment but cannot guarantee immediate interruption of a stalled GPU kernel.
Download-and-load timing includes network/cache effects. No inference cost claim
is made from CPU timing or free allocation alone.

All 64 observations form denominators, including invalid outputs. Schema validity,
target hit, wrong-control hit, blank miss and independently inspected one-selection
fixture completion are reported overall, by language and by family. Invalid outputs
are counted separately from wrong-target and blank misses. Wilson intervals are
descriptive because authored variants are correlated. There is no browser/native
execution, long-horizon recovery or production contract adoption in this experiment.
