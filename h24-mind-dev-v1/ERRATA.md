# H24 DEV V1 — diagnostic field naming correction

2026-10-18 (Asia/Riyadh).

## What was wrong?

The first one-off derived file `diagnostic-v1.json` mislabeled the per-case boolean `cases[].typed_valid`. Its implementation actually recorded **three-key field shape validity**, *not* strict Hala typed proposal validity, because the free-text `subgoal` identifier was not validated when populating that per-case field.

The file remains unchanged for historical provenance. In the same first diagnostic, the aggregate `total.strict_schema_valid=0` and the raw tool-choice and JSON syntax aggregate counts were correct; the original measured `result.json`, `verified-h24-dev-v1.json`, and model inference were not affected.

## Corrected independent evidence

Use **`diagnostic-v1.1.json`** for individual case analysis. Produced from the same unchanged original raw `result.json` through version-controlled Hala `tools/diagnose_h24_dev.py` with strict `tools/verify_h24_dev.py` validation and separate `json_syntax_valid`, `three_fields_valid`, `strict_typed_valid` booleans.

SHA256:
- Original immutable raw `result.json`: `a047f7652b1a954c745ba139431299602961cd017ad547253acea1b5fc24ef4f`
- Original immutable `diagnostic-v1.json`: `69bf1ca95be709e3aaef70b4ebaa71c4bf3e1541b37c971f760c584ba2128eac`
- Corrected `diagnostic-v1.1.json`: `ae1788145aca309dae66f05b93af27a4c9dbe1b560f0d44df07daa8618576bef`

Corrected counts on the same 20 family-isolated synthetic DEV examples:

| Metric | English | Arabic | Total |
| --- | ---: | ---: | ---: |
| JSON syntax parseable | 10/10 | 10/10 | 20/20 |
| Expected top-level three fields present | 10/10 | 10/10 | 20/20 |
| **Full strict Hala typed proposals** | **0/10** | **0/10** | **0/20** |
| Correct tool name **only**, even though proposal invalid | 3/10 | 3/10 | 6/20 |
| Proposed `visual` on a synthetic non-valid case | 5 | 5 | 10 |

No model suggestions were executed or authorized; the synthetic model/GUI task has no native Orcel executor. The 20 examples are insufficient to claim general computer-use safety or multilingual accuracy. These corrections do not change original metrics, training, dataset partitions, model or release status.

**Never delete, overwrite or silently relabel the earlier evidence.** No new Kaggle, GPU or training was required for this correction.
