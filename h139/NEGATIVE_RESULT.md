# H139 — independent React GUI transfer: NEGATIVE

Date: 2026-10-26. Scientific research record, not a release candidate.

## What was frozen before the independent test

- External pretrained point model: `patrickballou/gui-assist-grounder` (Qwen3.5-2B), pinned as defined by `h132.grounder_contract`.
- New H139 research rank-8 LoRA trained for exactly 64 optimizer steps using PRIVATE FREE Kaggle T4, on 64 source screenshots from two independent unmodified MIT GUI applications.
- Frozen adapter SHA-256:
  `c46ec0abf4d9fed545ec85867737ca19dc59f54f84e847b90304d4af6b052180`.
- Development split: 16 source-disjoint screenshots from those same two applications; independent owner Windows replay validated all 64 before/after raw model point responses.
- Truly independent split: eight bilingual screenshots from a third unmodified MIT React DnD application, source manifest SHA-256
  `c26c0b1c850bf4fc30c7e04bc16e547f89d69269174751f26ea2b6b824acfac9`.
  No source bbox or gold state is supplied in model input.
- Separate Kaggle blind inference run successfully completed **32 genuine neural START/END calls**, **zero optimizer steps**. All raw output types and native coordinate parsing were independently verified on owner Windows against the original held-out boxes; unchanged output and source SHAs qualified.

## Measured coordinates (not complete browser tasks)

| Metric | External pretrained Qwen3.5-2B | H139 research LoRA |
| --- | ---: | ---: |
| Same-app DEV START bbox hit (n=16) | 6/16 | 16/16 |
| Same-app DEV END bbox hit (n=16) | 7/16 | 16/16 |
| Independent blind app START bbox hit (n=8) | 4/8 | 6/8 |
| Independent blind app END bbox hit (n=8) | 8/8 | **0/8** |
| Independent blind app BOTH bbox hit (n=8) | 4/8 | **0/8** |
| Independent blind unique START predictions (n=8) | 4 | 2 |
| Independent blind unique END predictions (n=8) | 3 | **1** |

Per-language bbox hits on the independent heldout app (four EN / four AR):
- Pretrained START EN 0/4, AR 4/4; END EN 4/4, AR 4/4.
- H139 LoRA START EN 2/4, AR 4/4; END EN 0/4, AR 0/4.

**Inference:** The observed adapter greatly reduced destination-point diversity and failed on every blind END bbox. Improvements on familiar-layout DEV snapshots do NOT establish general visual grounding; the result is negative for independent-app destination transfer. This also cannot establish any Hala flagship 35B benefit because its weights are unchanged.

## Separate owner browser execution gate

The live browser-replay gate requires identical original Chromium screenshot SHA before executing model-proposed pointer actions, then scores only real persisted localStorage state. The original React cards have transient opacity animations, and initial owner replay attempts encountered screenshot SHA drift or navigation timing. A source-only independent Windows retry reproduced **7/8 frozen original screenshots with exact PNG byte SHA**; the first scene retained a small text-area difference and failed strict admission. These are **replay-qualification failures, not model-task failures**. Do not equate either frozen bbox hits or reference/oracle drags with actual model-completed browser tasks. Do not claim any full-task success until the independent replay evidence file is qualified.

## Next steps

- Keep original eight blind cases permanently evaluation-only. Do not add their original screenshot/labels or exposed response patterns to training, prompt tuning, checkpoint choice or repeated selection.
- Implement H140 on wholly new, legally usable sources with randomized **real** start/destination geometry, multiple independent apps, and new one-time unseen applications. Preserve Arabic/English balance, exact provenance, true browser task scoring and a coordinate-collapse gate.
- Keep all media, private weights, original geometry and raw model responses off public GitHub. Preserve upstream MIT attribution.
- No release, deployment, merge, paid cloud, dataset publication, flagship model promotion or product-level superiority claim.

Source: Draft PR #128 and H140 planning issue #129.
