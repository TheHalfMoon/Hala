# Hala H22 V4 — descriptive error analysis (frozen TEST)

**Status:** retrospective descriptive analysis of the already archived, unmodified TEST outputs; **NOT** a new training set, a new benchmark, an independent production security review, or permission to retune on TEST.

Source artifacts: the three SHA256-verified ZIP files in this directory. The independent `tools/verify_h22_batch.py` audit passed on the downloaded Saved Version. Reported assigned GPU UUID `GPU-a00cc5ee-b73b-753b-16e2-cdf6589573f1` was consistent across profiles; physical GPU identity was **not** independently attested.

## End-to-end synthetic task completion

| Precision | Completed | English | Arabic | Model errors | Unsafe proposals (evaluator count) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 2/8 | 2/4 | 0/4 | 0 | 0 |
| INT8 | 0/8 | 0/4 | 0/4 | 5 | 0 |
| NF4 | 2/8 | 2/4 | 0/4 | 24 | 12 |

These eight interdependent synthetic tasks per precision are not representative native Windows/Orcel tasks. The numeric difference in 2/8 versus 0/8 does not establish population accuracy, safety, or precision equivalence.

## HalaMind and action-level diagnostics

The following are recorded **step counts** within dependent task episodes; they must not be interpreted as independently sampled classification examples.

| Reported metric | FP16 | INT8 | NF4 |
| --- | ---: | ---: | ---: |
| English HalaMind correct / calls | 22/24 | 19/32 | 4/24 |
| Arabic HalaMind correct / calls | 4/32 | 4/32 | 24/32 |
| HalaMind invalid outputs | 0 | 0 | 4 |
| Correct plans / plan calls | 8/8 | 8/8 | 0/8 |
| Action errors / action calls | 42/55 | 49/59 | 17/44 |
| Model errors | 0 | 5 | 24 |
| Unsafe proposals (synthetic evaluator) | 0 | 0 | 12 |
| Unnecessary calls | 0 | 0 | 12 |
| Recovery events | 1 | 0 | 0 |

### Strong evidence of unresolved defects

**Arabic end-to-end failure:** all three precisions completed 0/4 Arabic episodes. FP16 and INT8 recorded only 4/32 correct Arabic Mind decisions each. This suggests the bilingual/RTL task workflow needs development on **new, separately split DEV data** before any future unseen TEST.

**NF4 safety/structure regression signal:** NF4 recorded 12 `unsafe_proposals` and 12 unnecessary calls, all in Arabic cases, with 0/8 correct plans and 24 model errors. These labels are evaluator counts, not evidence of actual unauthorized device actions. The Hala→Orcel boundary continues to treat all model output as untrusted and requires local policy approval and independently verified postconditions.

**Metric inconsistency cautions:** NF4 shows Arabic HalaMind step accuracy 24/32 despite 0/4 Arabic episode completion and 12 unsafe proposal labels, while English Mind was only 4/24. This highlights that intermediate step accuracy is not a substitute for verified final-state task success. Different downstream paths and dependent call counts make simple comparisons misleading.

**No precision adoption:** FP16 had zero reported unsafe proposals in these limited fixtures but still completed only 2/8 tasks; the measured safety boundary itself requires much broader adversarial and native testing. INT8 was not superior just because model files were smaller. No quantized runtime is promoted.

## Next independently justified experiments (prospective)

1. Freeze new Arabic and English application-family-isolated DEV fixtures with exact font shaping, RTL roles, normalized click targets, typed tool semantics, task descriptions, independent expected postconditions and synthetic image provenance. Do **not** train or tune on the 8 H22 TEST episodes or their output trajectories.
2. Qualify the unchanged HalaMind baseline on independently frozen unseen **planning**, ambiguity, tool selection and safe abstention cases; preserve Arabic/English breakdown.
3. Diagnose systematic failure classes on new DEV only: image-vs-accessibility grounding mismatches, coordinate protocol confusion, invalid structured responses, task-router mistakes, unsafe action proposal schema and final-state verification.
4. Consider a small separate LoRA or a stronger foundation only after a measured DEV baseline, legal/right-cleared bilingual data, valid checkpoint provenance and a post-training holdout plan.
5. Independently qualify Hala→**Orcel** protocol adapters and local approval/denial boundaries; do not let learned routing lower policy, mint receipts, or directly perform local OS actions.
6. Qualify H23 actual cancellation/recovery separately, with all its own genuine source, hardware and immutable receipt gates.

**Conclusion:** V4 closes the missing all-three-profile H22 batch evidence problem. It does **not** close the substantive model quality or safety problem, particularly on Arabic tasks.
