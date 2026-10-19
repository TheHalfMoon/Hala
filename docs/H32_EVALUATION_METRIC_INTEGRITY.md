# H32 evaluation metric integrity

H32 repairs source-only H28 analysis definitions. It changes no corpus, protocol, plan builder, raw evidence, model message, or evaluator plan.

## H28b paired-code metrics

`unresponsive_identical_code_pairs` counts only pairs where both outputs pass the existing strict C0–C7 syntax check and have equal raw codes. `pair_categories` partitions all 36 pairs into `both_valid_codes`, `contains_inference_error`, and `contains_invalid_output`. An inference error (`None`) takes precedence when a pair also contains malformed output. C0 is a valid withholding code and remains distinct from an inference error. Identical-code pairs are a subset of `both_valid_codes`.

## H28 replay metrics

`joint_task_language_state_rotation_instruction_language` reports 224 factor strata (2 task languages × 8 states × 7 rotations × 2 instruction languages), with 4 cases per stratum and 896 attempts overall. Each stratum includes format-valid output count, claimed inference-error count, semantic-correct count, raw wire-code distribution, and decoded semantic-code distribution. Malformed outputs and inference errors both appear as `INVALID` in the decoded semantic distribution; the wire distribution and `inference_errors` count distinguish them. They remain in case denominators and are never removed from correctness totals.

`wire_vs_semantic_c1_by_reference_state_rotation` reports each reference state and rotation with all attempted cases, a valid-code denominator, valid wire `C1` count, and valid decoded semantic `C1` count and rates. The valid-code denominator includes valid C0 withholding outputs. Malformed outputs and inference errors are excluded only from these conditional preference rates and remain visible in `cases` and overall metrics. This separates fixed wire-code preference from canonical semantic C1 selection.

All outputs remain deterministic CPU audit machinery for synthetic fixtures or externally supplied, hash-bound predictions. They do not establish actual model execution, causal effects, qualification, or permission.
