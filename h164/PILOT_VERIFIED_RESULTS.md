# H164 — independently verified first four-original-app TRAIN pilot

**2026-10-27. Source/data engineering result only.** No model weight training, no DEV or OOD benchmark, and no generalization claim.

## Exact owner Windows evidence

- New owner-private source `private-h164/four-app-train-pilot-v1/`, manifest SHA-256 `887eddd886d7f0505fcca4afd79ee96ce0dadf2443c72c0e5932f418e751d577`.
- **96/96 original Chromium source-native tasks succeeded**, 24 per original application: `tastejs/todomvc` (MIT, pinned `ff43b02e59dfa604386bb382034b2cd07c2bcd8a`); `react-dnd-kit-board` (MIT, pinned `553d4e92be3e86e0deb7045f4968152868399eae`); `arce/simple-table` (MIT asserted in upstream README, pinned `7580270d18fb764bbf437a5e5b88ea66d63fc449`); `ashishxa/calculator` (MIT asserted in upstream README, pinned `8f27a92072dde9658d8d94db8918a184ed7bd416`). Inspect full license terms before redistributing those last two projects; original sources remain local to their repositories.
- New TRAIN actions: **64 click, 24 native drag, 8 native keyboard type**. Bilingual tasks **48 EN / 48 AR**, **10 task families**. Screenshots **960×720**.
- The original native table's supported pageSize option was exercised at 8, 12 or 16 rows to vary visible layout while retaining unmodified source. Initial **56-image partial candidate was quarantined** after an independent duplicate-image SHA check; it has no manifest, never became training data and must never be imported. Fixed source collector successfully produced new 96 distinct images.
- Owner-independent `h164/audit_four_app_train.py` re-opened all 96 PNGs, recomputed exact hashes, checked actual typed native target points, validated application counts, ensured all are TRAIN and passed the `h164.protocol.validate_source_splits(..., admission="pilot")` no-leak gate. **144 previous consumed original benchmark screenshot SHAs excluded**.
- New independent browser contexts replayed **96/96 exact full PNG byte identity** (not perceptual similarity) with the original app's pre-action state. Owner-private QA receipt: `private-h164/H164_TRAIN_PILOT_PIXEL_QA.json`.

## New model format contract and promotion rules

`h164/native_action_contract.py` validates exact one-action JSON `click(point)`, `drag(start,end)`, `type(point,text)`, or `abstain`; rejects extra DOM/selector fields, nonfinite/out-of-frame coordinates, oversized text, multi-actions and gold on DEV/OOD. This is a contract and not proof that the frozen Qwen 2B can predict its schema.

**Do not train or promote on this pilot yet.** It meets only 96 of the H164 planned **384 fresh TRAIN** minimum; the new independent **96 DEV and 64 unseen-app OOD** datasets are still absent. Training on this pilot alone would violate the preregistered qualification plan. Expand new source coverage while keeping H163, H162, H140 and all other consumed evaluation records entirely outside optimization.

No GPU use, paid service, H140 sealed blind calls, Hala flagship 35B changes, public source/model/image release or main merge. Keep Draft.
