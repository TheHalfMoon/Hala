# H24 — HalaMind DEV prompt protocol V2 (prospective)

**Status:** CPU-qualified candidate under review, NOT measured model quality, NOT TEST, NOT production-qualified, NOT a native Orcel adapter. V1 raw and audit remain immutable in `ps/h24-dev-v1-evidence`.

## Why a separate protocol

The original personal-research Kaggle DEV V1 run (`f9bd2b011fb2f7dceed340e52d994a247888e833`) measured twenty syntactically valid JSON replies, but **zero strict typed proposals**: the unchanged base model wrote free-form explanatory English/Arabic text in `subgoal`. The independent corrected diagnostic measured only **6/20** correct *tool names* when extracted from schema-invalid responses (English 3/10, Arabic 3/10), which do NOT authorize any action.

V2 is a **prospective DEV prompt-format iteration**, not a new trained checkpoint or evidence that the original frozen V1 run was better. The model weights, Qwen3-VL-2B revision, dataset cases, application-family split, seed, device restriction, privacy and no-training rules stay fixed.

## V2 contract

- The versioned `--protocol v2` mode uses a distinct result schema `hala.h24.mind_dev_baseline.v2`, explicit `prompt_protocol=fixed_allowed_pairs_v2`, separate **private** Kaggle notebook identifier, and a separate independent auditor `--protocol v2`.
- Allowed typed `(tool,subgoal)` pairs are **only** `(visual,locate_verified_target)`, `(observe,refresh_stale_frame)`, `(observe,ignore_page_instruction)`, `(clarify,resolve_destination)`, and `(abstain,reject_external_command)`. All other pairs or free-form natural language subgoals **fail closed**; no suggestion is executed.
- HalaMind receives only the **DEV** user goal and trusted public-observation projection, not evaluator-only `protected` booleans, reference labels or held-out TEST examples. The V1 public projection remains the baseline; no hidden source label is added.
- Response validation independently checks JSON grammar, typed field/value restrictions, and exact ground-truth task decision. Failed model inference or any invalid pair does not count as success.
- Orcel remains the sole authority for permissions, execution and verified postconditions; no Orcel source is touched.

## Scientific limitations

The explicit five-pair vocabulary makes this an **instruction-format and deterministic routing** DEV experiment, NOT independent evidence of HalaMind reasoning or general computer-use proficiency. Telling the model what the possible subgoals are can improve schema adherence without improving task understanding. The 20 synthetic DEV examples have been used to motivate the change; they are therefore **adaptively reused development data**, not a fresh holdout. Do not use the 20 never-exposed H24 TEST labels or V1 raw predictions as training examples, and do not claim statistically meaningful Arabic performance from this alone.

## Qualification workflow

1. Keep V1 measured artifacts permanently unchanged. Run the V2-focused CPU tests, the complete Hala Python suite, genuine Jev, Graft and available Alibaba rule checks. A fresh-context PStack independent security review remains a separate unpassed gate.
2. Publish a **signed+DCO commit** on Draft Hala PR #5 and verify all exact-head GitHub CI checks. No merge.
3. Generate a fresh, private Kaggle notebook using `python tools/build_h24_kaggle.py --protocol v2 --source-commit COMMIT_SHA40 --output NEW_FOLDER`. The notebook may fetch model weights **on the free Kaggle T4 only**, never on the founder workstation.
4. After the remote Saved Version reaches a terminal state, download its raw result JSON, externally calculate SHA256 and run `python -m tools.verify_h24_dev PATH --protocol v2 --expected-sha256 RAW_SHA --source-commit SHA40 --output NEW_AUDIT`. Recheck all twenty original DEV rows independently.
5. Archive the raw results, error traces, exact commit/model/hash and the bilingual strict/relaxed metrics on GitHub in a separate immutable evidence branch; do not overwrite V1.
6. Decide whether further independent family/Arabic visual/closed-loop experiments are justified; **never adopt or release** solely on this DEV prompt-format score.

Training runs and production model adoption remain zero until separate founder/review/model-quality gates are satisfied.
