# H22 TEST-only restart — independent combined operator review

Scope: new TEST-only operator, qualification helper, draft amendment, builder handoff and focused operator tests. No production-phase source audit repeated. No GPU, inference, model download, network fetch, source edit or publication by reviewer.

Must fix: none. Worth considering: none for the fixed trusted-operator workflow. The amendment's deliberate GPU placeholder remains an execution blocker until actual qualification and external publication of the final manifest identity.

## Reviewed behavior

- Only local indices0..2 exist, fixed to fp16/int8/nf4 TEST. No DEV mode or automatic advance is reachable from the profile CLI. Frozen source07591, freeze98ca, original H19/model/font/retained-adapter/package/budget fields are preserved and checked against the original immutable SHA-bound study manifest. Clean complete Git source is checked before imports/launch as previously reviewed.
- First TEST requires all three fixed original DEV receipts and actual public raw ZIPs from original group `60a0986c...`, with pinned commits/receipt/archive hashes. Each passes the existing complete provenance/raw-result checks; confirmed DEV receipts enter the first new ZIP. The locally retained actual DEV ZIP bytes independently matched all three fixed archive hashes and satisfied these checks in the focused suite. Public retrieval itself occurs in the real gate, not this review.
- Later TEST requires the immediately prior new TEST receipt and actual ZIP under the distinct TEST path/SHA. Group/index checks prevent negative, boolean, out-of-range or wrong-group selection; DEV additionally requires its original SHA. Actual raw result hash/source/freeze/mode/split/status and terminal outcomes remain checked. Finished measured evidence additionally requires returncode0 and the experiment GPU UUID. No missing/forged raw-result bypass found in this delta.
- Qualification binds actual newly observed GPU UUID, real provider, original exact imports/bundle/font/retained archive and UUID-bound sampler without model loading. Placeholder UUID is rejected by amendment admission; qualification mismatch rejects before profile output/process creation. The root must publish the final postqualification amendment SHA and commit before inference.
- Each invocation launches one fresh bounded3700-second process, preserves terminal failure/timeout evidence, and exclusively creates profile directory/ZIP. Rerun prevention remains scoped to the trusted operator's retained working directory. Original DEV and new TEST remain separate physical-hardware groups; cross-group matched-device cost claims are explicitly unqualified.
- Operational reuse/copying is proportionate here: old immutable operators/manifests and DEV evidence are preserved, while separate paths/attempt identity avoid retroactively changing the completed study group.

## Independent checks

`python -m unittest discover -s outputs -p test_h22*operator.py -q`: 16 tests passed in0.408 seconds. Covers all3 old DEV prerequisites, actual retained DEV ZIP compatibility, TEST-only invocation/no overwrite, dirty source rejection, wrong GPU/missing DEV blocking before output, and missing/forged raw evidence. Five additional direct CPU cases confirmed wrong group, index0, boolean index, index4 and wrong DEV SHA reject before network.

Recomputed exact reviewed identities:

- TEST operator: `d8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e`.
- TEST qualification helper: `16d1c476ab0a5321c54d30838b4b7f96fc069990bc44efab11fedf994622e48a`.
- Draft amendment with blocking placeholder: `94b5a33358aa24c0335e5ea709aa707068f39d65c9697a56ca0c547b1a8e5f2f`.

Combined correctness/parsimony/security operator source verdict: clear. Final actual hardware qualification, immutable amended-manifest publication and per-profile durable retention remain execution gates; this is no new model-quality/adoption evidence.
