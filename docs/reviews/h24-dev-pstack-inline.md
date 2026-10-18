# H24 DEV runner — PStack-style inline review, not independent approval

**Date:** 2026-10-18. **Scope:** only Hala, branch `ps/h24-isolated-bilingual`, unpublished follow-up to Draft PR #5.

**Correctness:** New runner hard rejects non-DEV cases, validates exact frozen H24 source SHA, requires new remote output dir and exact checked-out Git SHA, scores raw schema responses independently, and preserves a language-balanced 20-case summary. The QA found an actual issue: evaluator-only `protected` labels were passed into the model prompt. This was fixed by constructing an explicit public observation projection; tests confirm labels and other oracle fields never enter prompts. These are **CPU tests only**, not a GPU model-performance qualification.

**Security:** No local model download, training, shell-action execution or Orcel edits permitted. Kaggle-only opt-in/preflight, immutable source/model revision, private notebook metadata, exact no-adoption flag. Untrusted page text is never made a trusted system instruction. Untrusted model output is limited to a typed proposal, with no execution grant. Remaining concern: remote notebook host/runtime and downloadable-model supply chain require actual attestation/monitoring when run; these source checks are insufficient as independent security approval.

**Product/science:** 20 synthetic, correlated DEV prompts are suitable for measuring this narrow parser/semantic router baseline, **not** for HalaType visual grounding, native Orcel closed-loop quality, or Arabic performance/generalization claims. FP16-only baseline; no LoRA, quant selection or fine-tuning performed. Holdout TEST and the historical H22 task trajectories remain unused.

**Parsimony:** Source-only remote benchmark and CPU evidence verifier preserve the existing H24 dataset and minimize new dependencies. A private Kaggle notebook can be generated only *after* exact reviewed commit and four CI checks pass.

**Actual gates:** 24 focused tests PASS; 303 full tests PASS with one skip. Graft build/check completed exit 0, deep semantic graph not qualified. Jev v1.13.0 completed six typed source/doc reviews (advisory). Alibaba Open Code Review delegation preview ran; **no real LLM-based Alibaba independent review was completed**. Fresh-context independent PStack/security review **unavailable**, leave Draft/merge and production gates blocked.

**Decision:** source preparation can be pushed to Draft PR #5 after signed DCO commit; prospective personal research Kaggle GPU must be separately submitted, and downloaded raw results verified. No training, release, merge, or Orcel changes are authorized by this note.
