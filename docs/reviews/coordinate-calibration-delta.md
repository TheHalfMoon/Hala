# Coordinate calibration delta review

**Verdict: pass for this experiment-only source delta; no actionable correctness, security, product or parsimony finding.** This does not qualify model or adapter quality.

Immutable source snapshot: `work/coordinate-calibration-review.diff`, SHA-256 `ce1b328fb6f099f5423d7e63a6fd78ed071e30c18b065f71bf1bdbad822549b3`. Scope: baseline parser/run metadata and prompts, four normalized-protocol tests, ADR 0003, regenerated notebook; supplemental calibration/smoke records checked below. No source edits by reviewers.

| Bar | Verified result |
|---|---|
| Correctness | `baseline.py:20–51` keeps strict integer-pixel decoding as the default. Explicit `normalized_1000` accepts an exact bare point object or a single exact JSON fence, rejects duplicate/nonfinite/extra/bool/out-of-range/ambiguous output, and deterministically floors each fixed 640×360 axis with the inclusive 1000 edge bounded to 639/359. No protocol guessing is added. Half-open target checking remains unchanged. |
| Correctness, metadata | `baseline.py:103–113,152–174` records calibration-v2 prompt revision, explicit coordinate protocol and `calibration_only`, keeps the raw answer, and stores decoded pixels separately. Pixel prompt/default remains unchanged. The regenerated notebook exactly embeds the canonical baseline, compiles, and has no stored output; its default invocation still uses pixels, so normalized calibration requires an explicit call. |
| Security | Bounded strict JSON decoding and fixed arithmetic introduce no executable tool, authority, remote fetch or code-evaluation path. Unknown protocols reject before runtime imports. Production contracts and the fail-closed real backend remain outside this experiment delta. |
| Product | ADR 0003 identifies reuse of observed fixtures as calibration rather than held-out qualification. The original strict result remains 0/8. The new calibration result is labeled and denies release qualification. A prompt/decoder change on these observed fixtures cannot establish general bilingual or complete-task performance. |
| Parsimony | One explicit mode parameter and deterministic conversion are proportionate to this diagnostic. No decoder inference, production adaptation layer or dependency is introduced. |

Verification: **11/11 relevant CPU tests passed** (baseline, normalized protocol, provider identity). The independent parser subreview passed 7 targeted tests and independently checked 6 valid edge/fence cases, 25 rejected mutations, 8 invalid modes and all 1,001 integer scale values. Reviewers used no remote jobs or local foundation weights.

Record verification: all eight calibration decoded-point and geometric-hit fields recompute from raw outputs. Recorded geometric hits are **7/8**, comprising **4/4 English and 3/4 Arabic**. Original/calibration IDs, goals, image digests and target boxes match exactly; these are the same observed fixtures. The original strict eight schema/hit rejections remain preserved. Both records deny release qualification.

The proposed smoke wrapper's legacy `verified_success_rate=0.875` matches the geometric calibration record and is explicitly bounded by `state_verification_scope=geometric_target_only_calibration`, a source-record reference, `release_qualified=false`, and a deficit narrative denying Arabic remediation or adapter-quality qualification. **No false complete-task-success claim found.** The two-step/rank-4/FP16 config consumes **two training microbatches** at accumulation 1; the artifact has 58 English rows whose processor inputs are audited before model loading. It does not train on every row, train Arabic images, or remediate the measured Arabic miss. No training run had been executed in the reviewed records.

Bound source/evidence hashes:

- Baseline: `a33495c574e0b6bebb9daab693db339c9d5340829605648838ff89b3bb13e02a`.
- Normalized tests: `8dbde783b3bc81f56fbf20ac4eac8b4988da3968697111e25c0d8d948973d45d`.
- Notebook: `f1cdd615d255bb26eab976c0137d0527e886b93d3e09e5c68e00126856b183d9`.
- ADR 0003: `05694b48c8e9ee03278103072f4768f26605b96eb60f99b6d0c6bdbda7ca7ce3`.
- Original strict result: `c2dd4f9b8820328e57ba5f8d96ab956228ea8d619869751813d9a6ae56f54253`.
- Calibration result: `a19eabe0d95d7c95ad4bf388758cf927f3f7daf5fcefaca6beb5f938d192982d`.
- Smoke baseline wrapper: `384305bf0994f7ad3bc5b87f6afbdd65ede576510b611814e6b2c10f489c7bc9`.
- Smoke config: `43a47a204212dcec79b8e9167c7afb3205d9378d20f46ae8fc4700fa56bf8911`.

Limits: the review inspected supplied experiment records and source; it did not execute or independently witness remote inference. No held-out evaluation, complete-task success, adapter training/quality, resumed numerical equivalence, quantization or production qualification is established. Any later two-microbatch run can support only the engineering behavior actually measured, with its data/language limits stated.

panel: narrow delta — correctness ✓ independent source/tests/record recomputation · security ✓ bounded parser/authority bar · parsimony ✓ combined inline bar · product ✓ explicit calibration scope; model/adapter quality qualification absent
