# H17: prespecified paired quantization experiment

Status: implemented CPU contracts; actual quantized GPU runs pending. No adoption decision.

Scope: NEW experimental files only; preserve all ten H14 frozen dependency hashes,
serving and trainer. Use the actual frozen64 FP16 baseline: schema-valid64, hit30,
wrong19, miss15; English16/32 and Arabic14/32. This is an insufficient production
baseline and dependent authored task sample.

Acceptance criteria:
- [x] Freeze source, protocol, exact library settings and baseline before any quantized inference.
- [x] Prespecify unchanged model/processor revision, SDPA, normalized H14 decoder,
  prompt, task order, media, seed42 and deterministic 48-token/15-second generation.
- [x] Separate opt-in remote Linux Colab T4 int8 and NF4 paths; reject local/Kaggle,
  external caches and offload; 1800-second wall deadline; no paid action.
- [x] Paired descriptive hit changes, per-case latency, language/family summaries;
  full-model and vision residency, quant-state tensor storage and allocator/device metrics.
- [x] Meaningful offline tests for mapping, corruption, config, pairs and pre-import guards.
- [ ] Independent root review and mechanical evidence accepted.
- [ ] Actual separate-process int8 and NF4 GPU results, including blocked outcomes;
  disclose reuse of the same verified free Colab GPU allocation and timing confounding.

Builder preparation hardstop: no model download or run before source gates.
After independent review and publication of the prospective freeze, the root
operator may run the authorized bounded experiment on a verified free Colab T4.
Experiment hardstops: no dataset/prompt tuning, native success/equal-accuracy/
adoption claim, local weights or paid infrastructure.
No open implementation question; remote runtime availability remains empirical.
