# Hala H24 — immutable Kaggle DEV V2 measurement

**Status:** real **measured but unqualified** personal-research Kaggle Saved Version, observed complete 2026-10-18. This is Hala-only; there were no changes to Orcel.

Kaggle: https://www.kaggle.com/code/abdulazizshehri/hala-h24-mind-dev-protocol-v2-20261018 (private)

## Precise source and experimental scope

- Published, signed and GitHub CI 4/4 source SHA: `a1256219b964e629f36ebfa8d9ae9aa7b79f2ecf` (Hala Draft PR #5).
- Unchanged `Qwen/Qwen3-VL-2B-Instruct` model revision `89644892e4d85e24eaac8bacfd4f463576704203`, FP16 on a **reported** Tesla T4. No adapter, training, model adoption, Orcel/native GUI execution or unrestricted system access.
- Twenty **reused, family-isolated synthetic H24 DEV cases**: 10 English and 10 Arabic, unchanged from H24 V1. Their adaptive reuse prevents interpreting V2 as fresh held-out performance. **H24 TEST not used**.
- Versioned `prompt_protocol=fixed_allowed_pairs_v2` and result schema `hala.h24.mind_dev_baseline.v2`. Five bounded `tool/subgoal` pairs were explicitly listed in the instruction. Inference stopped and original private result data were downloaded.
- `tools.verify_h24_dev --protocol v2` from the reviewed source independently verified the externally measured raw result SHA256, 20 case records, frozen family labels, pinned revision/source, strict recomputed scores and bilingual summary. The audit **cannot independently attest GPU hardware**.

## Actual unchanged measured outcomes

| Metric | English | Arabic | All |
| --- | ---: | ---: | ---: |
| Completed DEV raw responses (inference without exceptions) | 10/10 | 10/10 | 20/20 |
| Strict Hala typed proposals accepted by the original V2 evaluator | 0/10 | 0/10 | **0/20** |
| Strictly accepted correct decisions | 0/10 | 0/10 | **0/20** |

**Descriptive raw-output inspection (not a release gate):** all twenty outputs contained JSON that a standard parser can read; in **20/20**, `kind` was `"abstain"` instead of the required `"next"`. In nineteen, `tool` was also `"abstain"` and `subgoal` was `"reject_external_command"`; the other output lacked a qualified tool/subgoal. This conservative collapse occurred even on ordinary, authorized synthetic tasks. The **model did not adhere to the complete proposal schema**, despite the explicit V2 vocabulary.

The V1 run of these same DEV cases had **0/20 strict proposals** but `6/20` *descriptive* correct tool names extracted from otherwise invalid proposals. V2 did **not** prove an improvement in strict correctness. The small related DEV test set cannot measure actual agent performance or safety. Do not promote `"abstain"` as a valid schema-conforming `kind` merely to inflate accuracy.

## Exact evidence hashes (SHA256)

| File | SHA256 |
| --- | --- |
| `result.json` | `b473d21d4348b0d66a4e66aff64e9670e02502a0f7fe24d90e26cd0b257854e7` |
| `verified-h24-v2.json` | `82fd74118633b06984471475b7c33137cf5ebdd8b628ccd19b26fb8bd23b999c` |
| `notebook-source.ipynb` | `9162e3ee0b20bf6eeacb167c2e2d960cde783ab71781a66d643a8de7ec3501d0` |
| `kernel-metadata.json` | `0dde342b51b352b25e4cf82d11d6005e5b823697db2206420d58ffc759dd6f58` |

## Decision and next gates

1. **STOP re-prompting this same twenty-case DEV to chase scores.** It has already influenced the V2 protocol. Historical V1/V2 outcomes, case identities, and model checkpoints must remain unmodified.
2. Before further model tuning or inference experiments, independently freeze a **new and diverse application-family DEV** with authentic Arabic semantics and separately gated screenshot rendering. Exclude all H19/H22 TEST and H24 TEST cases and their trajectories. Include action semantics, multi-step postconditions and explicit abstention, beyond five templates.
3. Diagnose free-form instruction following vs over-abstention and measure valid proposals, safe refusals, wrong-target risk, bilingual accuracy, recovery and genuine task completion separately. Do not weaken the Orcel authority kernel, relabel `kind`, or bypass strict evaluator schema.
4. H23 actual CUDA cancellation/restart qualification and subsequent real closed-loop Orcel compatibility remain **separate unpassed gates**. No release, no merger of Draft PR #5, no native device executor, no paid compute.

Retain all V1 and V2 raw files append-only. Successful remote execution and SHA-matched audits establish preservation of **unqualified experimental evidence**, not safe/successful agent behavior.
