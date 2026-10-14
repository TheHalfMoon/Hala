# H14 independent combined review

Reviewed snapshot SHA-256: `5c549e2d1e1b8c080a8fcef20033cd604241ca0a047b0b735d7c08cea3f152c9`.
Trusted pre-inference manifest SHA-256: `b2488df604bd16380ad72f80734fcf8b5e38ce873f54e544ecefcf2f81eb5a87`.
Fresh-context source review of the eight inventory files, frozen dataset/manifest, applicable repository instructions, product/specification, earlier calibration protocol and training-family generator. Inventory hashes and every manifest member independently verified against the supplied hashes. No source edits, weights, GPU allocation, model inference, commit or push performed.

## Must-fix

- **Correctness [P2], evaluation/heldout_runner.py:72-73:** Each persistence call opens the sole `result.json` in truncating write mode. A process/runtime loss or write failure during an update can destroy all previously persisted observations; the final persistence call uses the same unsafe operation. This undermines the advertised inspectable partial observations on a scarce remote run. Write canonical bytes to a same-directory temporary file and atomically replace the destination after a completed write; add an interruption/write-failure test proving the previous complete result survives. Use the same small helper for image-freeze provenance. This is a CPU-verifiable retention fix, not model qualification.

## Worth considering

- **Correctness, evaluation/heldout_runner.py:139-141:** `preprocessing_seconds` is sampled before CUDA synchronization, although `inputs.to('cuda:0')` can enqueue asynchronous transfers. Move its sampling after the barrier to include transfer completion. End-to-end `call_seconds` already covers the barrier; this affects the named preprocessing breakdown rather than hit scores.
- **Product/evidence, tests/test_heldout.py:123-134:** The freeze test compares annotations to the same authoring implementation and checks old goals/IDs are absent, but does not independently derive expected answers from displayed values. Existing manual source inspection supports the current annotations; add small semantic checks for earliest arrival, minimum stock, maximum demand/latest version and unique urgent-review card if this generator will evolve. This is useful evidence against future label drift, not a present annotation defect.

## Skip-it / clean

- Strict parser accepts only bare JSON or the explicitly calibrated single LF `json` fence; rejects duplicate/additional keys, booleans, floats and nonfinite values. Endpoint mapping and half-open target geometry match the documented DPI-independent image frame.
- Prompt allowlist passes goal and protocol only; renderer reads visible content and boxes but not private expected target/terminal state. Runner supplies the image and that prompt, not the case object, to the processor/model.
- All eight authored semantic families differ from the prior save/open/archive/export/toggle/form/filter/drag-drop/recovery/scroll/rename/delete workflows and the old four-button calibration. Eight variants per family remain correlated; the documentation and summaries explicitly disclose that limitation rather than treating them as representative native tasks.
- Target annotations are consistent with visible identity/value semantics on source inspection, including transit hour rollover and Kanban review-column distractors. No old task/image reuse found in the scoped implementation.
- Arabic rendering fails closed without actual RAQM; control-label clipping guard measures all lines. The local English renderer tests are real Pillow operations. Arabic render/model success remains unverified on this CPU environment, which lacks RAQM.
- All images and font/render provenance are persisted before model/processor loading; PNG hashes are rechecked before each call. Immutable model revision, package pins, FP16/SDPA settings and memory accounting are explicit.
- Negative controls fail wrong-target, invalid and false-completion cases. Completion is exactly the declared one-selection synthetic state transition; separate documentation prevents interpreting this as native or general task qualification.
- Summary denominators include invalid outputs. Wrong-target and blank miss exclude invalid schema intentionally. Wilson intervals are descriptive, with language and family summaries and dependence disclosure.
- Outer 1800-second alarm and generation token/time limits are explicit; docs correctly explain that a GPU stall may resist immediate interruption. Failed/partial runs do not receive a full-run score.
- Parsimony: a small independent experimental protocol/fixture/runner is appropriate. No framework, speculative abstraction or production-contract change warrants removal.

## Verification and verdict

Independent bundled-Python CPU execution: `python -m unittest discover -s tests -v` completed **105 tests, OK**, in 8.408 seconds. This includes 14 H14 tests and actual English rendering/half-open border checks. Python 3.12.14/Pillow 12.3 CPU execution does not validate the separately pinned remote ML runtime. No model score or release qualification is inferred.

Verdict: **fix the persistence retention issue before the remote experiment; otherwise the scoped CPU implementation matches the prospective specification.** Actual bilingual GPU evaluation, Jev/Graft evidence and security review remain separate parent-owned gates.

panel: full - correctness/parsimony/product: fresh-context combined Codex judge (inherited session model); security: separate judge, outside this report; no d1 substitution.
