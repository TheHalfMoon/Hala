# H25 projection V2: separate source-only input admission correction

**Status: research preparation only.** Hala PR #6 already froze H25 source (`3c87c4b502ab2497bdf02dec28a35cd11f7bab8f`). This new, separately stacked branch does **not** edit its immutable `h25/frozen/*` cases, H24 historical traces, H22 TEST, or any other repository. It does not train or execute Hala or Orcel and does not qualify a TEST benchmark.

## Concrete audit findings and repairs

1. **Oracle-friendly state flags in original H25 DEV output.** The original `public-dev.jsonl` `model_input.observation` carried `required_parameter_present`, `conflicting_evidence_present`, and `postcondition_verified` booleans. These directly encoded parts of the expected decision and could overstate reasoning accuracy. The V2 input has only two frame IDs, two synthetic control labels/bounds, and untrusted page text. The case's own natural-language goal remains exactly as originally authored, unmodified. The model must interpret whether to clarify, reobserve, abstain or verify.
2. **Semantic exposure in frame IDs.** Original H25 `frame_id` contains `f-<family>-<rank>-<language>`, and stale-frame IDs start with literal `stale-`. Family and case rank identify the source and can become a trivial classifier in a publicly known set. V2 replaces both with deterministic, uniform-length hex tokens; the *equality relation*, essential for detecting staleness, is preserved. These are pseudonyms, **not secrets**: public source cases can still be correlated with the published projection. Do not call H25 an unseen benchmark.
3. **Uninformative fake UI label.** For each family, the old synthetic target button was simply the first word of the first task (e.g. `Find`), with no target identity or task context. V2 substitutes distinctive English/Arabic, family-specific synthetic control labels and an archive/history distractor. This remains synthetic **text-based UI metadata**, not a real browser snapshot, OCR/RAQM-qualified visual grounding, or permission policy.
4. **Evaluator metadata boundary.** V2 keeps `source_ref` completely outside `model_input`. A future runner must only submit `model_input`, never the containing object. Case references and private oracle labels are still available **only in the separate unchanged frozen H25 source** for an external evaluator. No `expected`, `state`, `partition`, `unsafe_user_intent` or previous model predictions are supplied as LLM input.

## Reproducible candidate identity

- Original first-party H25 source archive SHA256: `b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b` (**unchanged**).
- New 42-case bilingual DEV projection (21 English/21 Arabic, three new families) deterministic canonical SHA256: `cceed2575f3f01995d489c811e2ee38f32272492fc2b533c0197ec0ba27c8ccd`.
- Prospective `manifest.json` SHA256: `9668119f01297e7ba81e33ddf900aedf2acb1ef88ffe2684321df811cc89a280`.
- Prospective `model-inputs.jsonl` SHA256: `6d4d18d68c5bbf4646d121781dc0fed8dad3cd3fafc017ebd6572eeac6f9301d`.
- Generator: `python -m h25.projection_v2 --output NEW_NONEXISTENT_FOLDER`. It rejects overwrite and verifies original H25 source before generating. The three exact output digest checks live in `tests/test_h25_projection_pins.py`, and negative tests in `tests/test_h25_projection_v2.py`. The source and digest-locked regeneration are sufficient to validate this prospective projection even if a standalone output folder is not uploaded.
- None of these digests establishes real-world security, model quality or validated Arabic semantics.

## Bilingual editorial review, still provisional

A side-by-side first-pass reading of the 42 source language pairs across six domains found the English/Arabic goals generally preserve their requested task type (calibration, donation inventory, museum accessions, seminars, accessibility, wetlands). This is **the same assistant performing an inline reading**, not a separately recruited Arabic native-speaker reviewer, and does not certify translation equivalence.

Important cases still needing independent adjudication include missing/ambiguous parameters, contradictory signed evidence, the policy for hostile **page** instructions vs destructive **user** instructions, and confirmation of persisted postconditions after an apparent Save/Publish. The original `route()` precedence was not changed; its source-based labels are synthetic and do not establish authority for Orcel.

## Explicit next gates

- Keep new DEV public/non-holdout; do **not** call it a blind final TEST, especially after publishing on GitHub.
- Obtain independent bilingual reviewer decisions and rights/legal consent if the data might be used to train a model. Original H25 marks `commercial_training_clearance=review_required`.
- Require a separately approved versioned decision schema for the two H25 subgoals absent from H24 V2 (postcondition verification and conflicting evidence clarification) before evaluating a model. Do not silently coerce H25 outcomes into the five H24 V2 choices.
- Ensure model output is **never directly executed**. Hala is a proposal model only; Orcel alone evaluates authorization and independently verifies real device postconditions. Do not edit Orcel from this scope.
- H23 real CUDA cancellation/restart remains unrun/unqualified. No GPU inference/training, paid compute, deployment, model adoption, PR merge or release is authorized by this projection change.
