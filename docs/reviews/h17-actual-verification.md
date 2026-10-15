# Independent H17 actual-evidence verification

2048 independent assertions passed. NF4 included; both measured modes reviewed.

Recomputed strict JSON schema (including duplicate-key rejection), normalized integer-to-pixel mapping, first containing control, expected-target hit/wrong/miss and synthetic selection completion for every raw output. Recomputed every overall/language/family count, rate and descriptive Wilson interval, paired case and group statistic, latency quantile, memory arithmetic and recorded tensor-storage sum. No project scoring or paired helper was imported. Frozen H14 and H17 file hashes match the prospective manifests.

| Mode | Valid | Hit | Wrong | Miss | EN hit | AR hit | Warm median / p95 call seconds | Allocator peak allocated | Max sampled device used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 64 | 30 | 19 | 15 | 16/32 | 14/32 | 1.383885 / 1.897197 | 4587264512 B | 4792909824 B |
| int8 | 64 | 34 | 20 | 10 | 20/32 | 14/32 | 3.961039 / 5.843959 | 2775859712 B | 3196977152 B |
| nf4 | 64 | 32 | 21 | 11 | 19/32 | 13/32 | 1.554220 / 1.700131 | 1909077504 B | 2140012544 B |

int8: pairs {"both_hit": 28, "lost_hit": 2, "gained_hit": 6, "neither_hit": 28}; hit Wilson 95% descriptive [0.41072087051471007, 0.6482401257482395]. Full final recorded storage 2446402112 B, vision 412244032 B; 1228 tensor records on cuda:0. Quantized modules 300, including 104 vision modules. Final dtype storage: {"torch.float16": 631185408, "torch.float32": 3277376, "torch.int8": 1811939328}.


nf4: pairs {"both_hit": 26, "lost_hit": 4, "gained_hit": 6, "neither_hit": 28}; hit Wilson 95% descriptive [0.3810209474394701, 0.6189790525605299]. Full final recorded storage 1565929792 B, vision 216331840 B; 1828 tensor records on cuda:0. Quantized modules 300, including 104 vision modules. Final dtype storage: {"torch.float16": 631185408, "torch.float32": 463168, "torch.uint8": 934281216}.


Source freeze SHA256: d9cbb88f567e43325028151d39c1a00e2ae9c9081fdd1556094aedce8e69585c. Operator-reported actual source commit: f5f6f34ef51e71b16d8ea5c293161991cb4eba8f. Artifact SHA256: fp16 6a1adf9a4dd322c06e07ed317f1d880763c60c214d1cc87c036dea21567a7a32, int8 4ed6c689b411c27fb0b935bc4bbdeb020ec2513a5df90bdb1baff43fa968955d, nf4 42c9d4a777926093cbd99bb840200314f10b7bcc3ac5f09ec8f89fb9cfab0b20.

The frozen runner loads the unchanged base model directly, without an adapter-loading path. The operator reports one fresh subprocess per quantized mode, reusing the same verified free Colab T4 allocation after the preceding model process exited. Result JSON alone does not independently attest process isolation, billing status or GPU allocation history.

The JSON companion contains all independently recalculated family/language summaries, pairs, latency and residency aggregates. Quantiles use linear interpolation at (n-1)p; warm excludes the first call. Full/vision storage is the sum of recorded unique-storage sizes, including exposed quantization state. The logs omit storage pointers, so independent deduplication or discovery of unrecorded tensors is impossible; this verifies the recorded inventory, not a fresh live-model traversal.

Allocator peak allocated/reserved and maximum sampled whole-device usage are different measurements; device sampling is not a continuous device peak. Separate FP16 and quantized runs/GPU allocations and sequential mode subprocesses confound latency; no controlled speedup inference. Eight authored families contain dependent variants: Wilson intervals are descriptive, not representative population confidence. Equal or higher observed counts do not establish equivalence, general improvement, representative bilingual ability, native task success, commercial cost, release or adoption qualification. Synthetic fixture completion is exactly the selected-target transition. No model was run or downloaded during this review.
