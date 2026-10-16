# Independent actual H21 development baseline review

**Recorded experiment integrity passes; strict development integration quality is poor.** Independently reran the standard-library audit: **1,124 checks passed, zero errors**. Result SHA256 `e9f98694089d5411d3a6c570c7de2ede8f82cbeb3132e47904d17301030c0768`; externally archived freeze `ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e`. An initial invocation mistyped the manifest argument and correctly rejected it; the successful rerun used the coordinating session's independently archived pin, not a digest inferred from untrusted manifest contents.

## Actual outcomes

| Measurement | English | Arabic | Total |
|---|---:|---:|---:|
| Development episodes completed |0/4|0/4|0/8|
| Development Mind semantic correct |0/32|0/32|0/64|
| Initial plans semantic correct |0/4|0/4|0/8|
| Standalone authored test Mind exact correct |0/5|0/5|0/10|
| Strict Type proposal rejections |2|4|6|
| Development model calls |34|36|70|

All eight actual final snapshots retain revision0, original current/saved field values and original current/saved card locations. Raw-call/trace bindings and independent persisted-state checks confirm **no edit, movement or save occurred**. There were no reached transient-save failures, recoveries or redundant observations; zero recovery counts are not evidence of recovery capability. The global log contains80 calls:70 development calls plus10 separately labelled standalone Mind diagnostics. No held-out GUI episode test split is included in this reviewed result.

## Bounded development-only forensic classification

For the64 development Mind calls: **30 invalid protocol outputs,28 syntactically valid responses of the wrong requested kind,6 valid-kind but semantically wrong responses,0 semantic correct**. This decomposition uses raw development calls and the unchanged strict protocol/public goal semantics. Standalone test diagnostics were only checked against their frozen labels and counted; no test-specific error analysis or tuning was performed.

**Correct the interpretation of `unsafe_proposals=6`: it counts six protocol-rejected Type outputs, not six out-of-frame or external-action attempts.** Four are fenced JSON click objects; their diagnostic coordinate pairs `[500,148]`, `[319,245]` (twice), `[688,445]` are all within0..1000. Two are truncated/malformed fenced echoes of public input, not complete executable action objects. Diagnostic fence removal was used solely to describe the raw development failure; these objects remain rejected and were never rescored, executed or adopted. No confirmed out-of-range proposal appears among the four interpretable coordinate bodies. The two malformed echoes do not support coordinate-quality claims. Their parser rejection leaves state unchanged. There is no basis to label these six as attempted native/external malicious operations.

This is a **strict protocol and task-integration deficit**, not proof that the unchanged base is incapable of planning, grounding or Arabic generally. Preserve the fixed prompt/data/parser, strict scores and predeclared pilot schedule; this report proposes no repair, rescore or test-driven schedule change.

## Artifacts and source

- Independently checked all **six actual PNG byte hashes**,800x600 IHDR dimensions, local episode media hashes and matching Type global-call image hashes. Each also matches its prospectively frozen media record. The six actual frames are retained; no screenshot pixels were substituted. This check does not inspect glyph shaping or independently assess visual grounding quality.
- Frozen dataset/settings/media/training-row file hashes and frozen source-map identity pass the independent audit. The recorded runtime SHA `da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35` matches the runtime file bytes at commit `6b9bdff139d57ca29d0e4c5774c2d71805e8188e`. Live GitHub commit query independently confirms that commit is published and signature-verified valid.
- Independently checked retained ZIP **93,929 bytes**, SHA `4542c3497b632be1ca261f75e3136cc68e18b9a6468ff62eea49f04d596957e2`,12 entries, CRC valid. It preserves result/log/launcher/package evidence, six actual frames and the initial failed-import evidence. The first failed attempt remains distinct from this measured run; it produced no model-call score.

## Recorded latency and memory

Actual metadata reports TeslaT4, CUDA13.0/SM7.5, Torch2.11.0+cu130, Transformers5.16.1, Pillow11.3.0. Load86.585s; total wall332.570s; process CPU264.476s; summed model generation232.169s. Independently summing per-call generation matches the reported total.

Measured callback wall medians/P95 (nearest rank): development Mind64 calls **2.560/4.469s**; Type6 calls **3.421/8.680s**. The first global call is4.455s. All79 subsequent mixed-component calls have median/P95 **2.557/6.289s**; this is an observed sequential sample, not throughput or a production SLA. Type's P95 is its maximum because n=6. Load/cache/kernel warmup and long malformed completions influence these timings.

Recorded CUDA peak allocated **4,514,018,816 bytes**, peak reserved **4,575,985,664**, maximum sampled whole-device use **4,744,675,328**. Recorded process peak RSS **7,793,696,768 bytes**. Independently aggregated before/after/load/final samples reproduce the maximum device-use field. Allocator peaks, sampled whole-device usage and process RSS measure different quantities; full-device includes driver/other contexts, and samples are not continuous device peaks. These records do not establish production serving cost or independently attest hardware behavior.

Additional replay/forensic evidence: `work/h21-base-dev-review-audit.json`, `work/h21-base-dev-actual-review.json`; reproducible supplemental script `work/h21_dev_actual_review_probe.py`. No published scoring functions were imported; no repository source, prompt, data, parser, schedule, model or GPU state was changed by this reviewer. This review neither performs nor claims the held-out episode test, training, useful adapter persistence, quality improvement or release qualification.
