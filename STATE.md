Updated 2026-10-14, branch ps/foundation.

Signed bootstrap main 00f6692; draft PR #1 published. CPU/source/helper foundations and independent repairs reviewed. GitHub CI 3.11/3.12 passed at dbce84e; 91 local CPU tests pass after new provider/protocol regressions.

Actual free Colab T4 finally assigned; tensor probe and unchanged Qwen3-VL-2B FP16 ran. Strict pixel-JSON result 0/8 preserved. Explicit normalized calibration 7/8 (English 4/4, Arabic 3/4) on the same eight fixtures; not held-out or complete-task qualification. Model allocator about 4.27 GB, peak about 4.35 GB. Weights remain remote. Provider/font fixes reviewed; coordinate delta independently reviewed, no findings. JSON/reports under experiments/runtime and docs/reviews.

Next: verify final CI; freeze protocol and evaluate held-out tasks; actual two-step English processor/LoRA/checkpoint engineering smoke (inputs proposed only; audits 58 rows, consumes two microbatches); justified adaptation/quantization. No real adapter, quantized model or native Aladdin integration exists yet. English engineering training would not claim to fix the observed Arabic miss.

Remaining limits: Arabic training shaping/external rights; native integration/release evidence; no measured full-task success or production operating cost. No paid resource, merge or production launch activated. Aladdin read only.
