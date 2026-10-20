# H35 — H34 retained bilingual visual-grounding failure forensics

**Status:** Draft, SHA-bound post-hoc analysis of original H34 synthetic DEV. No model training or inference, no new GPU run, no native action, no independent TEST, no qualified product release.

## Retained evidence, NOT new measurement

Replays the original H34 audit, using its fail-closed ordered raw records, per-image SHA256/PNG checks, frozen corpus and independent expected raw digest. Both original Kaggle runs were on the same 32 first-party synthetic DEV cases; neither is a blind independent test.

- **Original run** raw SHA256: `6141780732186bb4503467dc27a44169576cbf5ab2a3f1c41ee163669da6956c`; all 32 are `parse_invalid` (Markdown fenced output).
- **Revised-prompt run** raw SHA256: `f9dfbba4a4d8adee149caa0f5b2c18def8b9383f472c0fc1207d171996809bcf`; all 32 parse, **8 intended-target hits**, **2 wrong-control bindings**, **22 coordinates outside every observed control**, zero abstentions.
- Frozen H34 case SHA256: `4b7f849a624caac23dd857adb19ed4293d93d0693dbdbd2921d4423b666a6a3a`.
- Derived, deterministic per-case and stratified post-hoc report `docs/reviews/h35-h34-v2-grounding-failures.json`, SHA256 `b704c9062af80357490d5b2ff8c92986a704053b1be9a2ebaff9e7d1f4780d23`. This derived report never overrides the original raw receipt or H34 scoring. 32-case target labels remain evaluator-only.

## Stratified confirmed counts on revised-prompt synthetic DEV

| Axis | Segment | Correct / total | Wrong eligible target | Point outside all controls |
|---|---|---:|---:|---:|
| Layout | row | 1/8 | 0 | 7 |
| Layout | column | **0/8** | 1 | 7 |
| Layout | rtl_row | 4/8 | 1 | 3 |
| Layout | scattered | 3/8 | 0 | 5 |
| Language | English | 3/16 | 1 | 12 |
| Language | Arabic | 5/16 | 1 | 10 |
| Domain | mail | 3/8 | 0 | 5 |
| Domain | calendar | 1/8 | 0 | 7 |
| Domain | files | 3/8 | 0 | 5 |
| Domain | shop | 1/8 | 2 | 5 |

These are **descriptive**, tiny-sample, post-hoc results. Do not infer the model is inherently better at Arabic, worse at columns in general, or that adjusting coordinates will fix it. Category-level uncertainty is substantial and external validity untested. The qualitative observation is that valid format does **not** entail usable pixel grounding and layout-dependent coordinates are a high-priority hypothesis for the **next prospective, independently designed** experiment.

## How to rerun offline

```bash
python -m h35.forensics --evidence reviews/h34-kaggle-run-20261020-v2 --expected-raw-sha256 f9dfbba4a4d8adee149caa0f5b2c18def8b9383f472c0fc1207d171996809bcf
python -m unittest tests.test_h35_grounding_forensics -v
```

**Verified Windows source checks:** H35 focused 8/8 tests passed; full Hala suite **487 tests run / 485 passed / 2 skipped**; compileall PASS. Actual Graft wiring-only build/check PASS with **1,511 nodes / 3,942 edges**; deep semantic tier not run. Independent model-driven PStack, Alibaba and D1 security-review receipts are not claimed, and exact-head GitHub CI is pending until the PR is pushed.

No data modification, external network, Kaggle allocation, GPU use, credentials or Aladdin modifications. The analyzer separates `inference_error`, `not_attempted`, `parse_invalid`, `abstain`, `off_controls`, `overlapping_controls`, `wrong_control`, and `hit`. It refuses to score changed raw evidence under a previously retained SHA and independently cross-checks classifications with the original H34 boolean score.

## What H35 does **not** solve

No tested improvement in neural visual grounding, calibrated abstention, real Arabic text understanding or production usability. Do **not** adopt an unqualified model, change the original H34 raw data/evaluator, treat these cases as new TEST, or automatically snap off-target points to nearby controls. Any new method requires a prospective protocol, separately pinned inputs, actual model inference and held-out tests.

Next: design and execute independent controlled model/geometry improvement experiments using only verified permitted no-cost resources; qualify real captured RTL UI, H23 real CUDA lifecycle, independent Arabic adjudication, code reviews, actual model performance and founder release gates. Keep host action permission deterministic and fail-closed.
