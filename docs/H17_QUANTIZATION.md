# H17 paired experimental comparison

The measured H14 FP16 baseline has 30/64 hits (English16/32, Arabic14/32),
64 schema-valid outputs, 19 wrong-target selections and 15 misses. It does not
qualify production, general bilingual competence or native complete-task success.
H17 never turns equal observed hit counts into an accuracy-equivalence or adoption
claim. Eight authored families have dependent variants. Reusing this exposed set
is a paired diagnostic, not a new blinded qualification set. Do not tune on it.

Both modes use the exact pinned model/processor revision and unchanged H14 functions.
The existing actual FP16 result is included byte-for-byte; supply its original frozen
PNG directory, not newly rendered approximations. PNG/RGB/dimension hashes are checked.
int8 uses threshold6 and no FP32 CPU offload; NF4 uses double quantization, uint8
storage and FP16 compute. The intended scope includes all transformable linear modules,
including vision; record actual converted module names and full-model/vision storage.
Unconverted vision tensors stay FP16. This is mixed precision, not an assertion that
every tensor has the requested bit width. No explicit skip list or tuning is allowed.

Official evidence is captured with URL, SHA256 and relevant exact-version source
excerpts in official-sources.json. Runtime pins are torch2.11.0+cu130,
transformers5.16.1, Pillow11.3.0, bitsandbytes0.50.2 and accelerate1.12.0.
PyPI runtime constraints admit these versions; Transformers' tagged integrations
support BitsAndBytesConfig and CUDA device maps. bitsandbytes' optional test extra
still constrains transformers<5: that extra is not installed and does not prove
runtime incompatibility. The compatibility of this exact multimodal path remains
GPU-unqualified until actual execution. Do not downgrade to obtain a result.

Only a fresh model subprocess in a verified free Linux Colab T4 runtime with
CUDA13.0 and explicit opt-in is allowed. The existing free GPU allocation may be
reused after preceding model processes exit; this reuse must be disclosed.
Clear inherited cache variables or point them inside /tmp/hala-h17-cache before run.
All optional framework imports and weight access occur after remote/cache/source/media
validation. Do not mount Drive or use a persistent model cache. Model/cache cleanup
is the notebook owner's responsibility after exporting results. The runner retains
blocked/partial evidence and never produces a complete paired score for a partial run.
Run each mode in its own fresh subprocess, using the same frozen checkout and media:

```
export HALA_REMOTE_QUANT_ALLOWED=1
python -m evaluation.quantization_runner --mode int8 --images /tmp/h14-images --output /tmp/h17-int8 --source-freeze-sha256 FROZEN_SHA256
python -m evaluation.quantization_runner --mode nf4 --images /tmp/h14-images --output /tmp/h17-nf4 --source-freeze-sha256 FROZEN_SHA256
```

These are prospective commands, not actions performed or GPU qualification evidence.
Install exact pins only in the authorized remote session. No notebook provisioning,
paid action, weights on the Windows PC or Kaggle use is authorized by this artifact.
Wall, CPU and load times, actual device/allocator residency and cold/warm per-case
latency are observations. Baseline/candidate sessions are temporally separate:
latency deltas have environmental confounding. Free-session observations do not
establish commercial serving cost. CPU/fake paths validate contracts only.
