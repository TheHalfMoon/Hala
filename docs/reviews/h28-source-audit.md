# H28 inline source review and qualification boundaries

**Scope:** new H28 prospective DEV package and its CPU-only plan/auditor. Internal source review, **not** independent security, PStack, Alibaba-model, Jev, held-out TEST, or production certification.

## Source and safety checks actually performed

- `h28/corpus.py`: 64 newly-authored synthetic bilingual first-party public DEV scenarios, eight states per domain including deliberately uncertain permission. Pinned SHA256 `63f67ad16d23fe19c178c6dd35ef2d06ad548c691be514d2b694b8440bd5955c`. Exact dataset reconstruction and case language/family/state distribution checks. These are not independently translated or visually reviewed and contain expected-state clues as plain text.
- `h28/protocol.py`: seven invertible cyclic C1–C7 label rotations with C0 fixed, two instruction languages. Only `model_input` crosses into the model messages. The observed page-injection content appears inside the user JSON only, not a privileged system message. Output maps to original `h26.choice.advisory` typed, **untrusted**, non-executable proposals; native authorization is not exposed.
- `tools/build_h28_dev_plan.py`: create-only CPU plan builder, 896 planned messages, separate evaluator sidecar, hash-bound manifest, no GPU/model/network calls. Request IDs are sidecar metadata; sending whole envelope to the model would invalidate experiment claims.
- `h28/replay.py`: externally given raw outcome file must have exact externally observed SHA256, finite regular size, strict JSON model/source/plan fields, 896 ordered keyed replies with typed error-vs-raw discrimination; labels and rotations are independently source-recomputed. This proves internal consistency of submitted bytes, **not** that a GPU or model actually generated them.
- New H28 unit fixtures intentionally include synthetic always-withhold, always-C1 and perfect-oracle cases. None is empirical accuracy evidence. `permission_uncertain` C0 means withholding rather than granted authority; the access-like statistic counts visual target proposals, not all refusals/clarifications. C7 on an inappropriate benign request remains semantically incorrect.
- **Actual local checks:** 443 Python tests run, **442 passed / 1 skipped**; 21 H28 targeted tests pass; Python compileall PASS. Genuine Graft non-LLM build/check PASS: **1,368 nodes / 3,590 edges**, meaning tier intentionally not built (no paid provider). Genuine Alibaba Open Code Review **preview only**, not LLM-reviewed, reviewed-file filtering excludes tests by default. No independent PStack, Jev H28 advisory, vendor-model code review or real E2E review has occurred.

## Open findings / risks

1. **No model-measured H28 result** or corresponding free-qualified Kaggle worker and pinned model-revision run; 896 means prospective matrix cells, not GPU calls.
2. Original bilingual semantic labels are authored by the same developer. Native Arabic expertise, permission/policy validity and adversarial independently labeled split are not established.
3. Public prompts/cases are not genuinely hidden unseen TEST. Their own stated states and symbolic frames may lead to unrealistically easy or misleading model scores.
4. Text-only code-choice correctness is not screenshots, host permission checks, OS operation prevention proof, postcondition verification or complete-task success. Real HalaType RTL screenshots and H23 CUDA in-flight cancellation/recovery are not qualified.
5. Model-claimed provenance or provider fields in a JSON outcome cannot independently attest actual model execution, hardware ID, rights, costs or inference authenticity; preserve independent original provider artifacts if a lawful future run occurs.
6. Kaggle no-cost and non-commercial-use restrictions remain a real usage qualification blocker until evaluated; no paid GPU, billable overflow or large Windows weights allowed.
7. Founder controls Hala main merge and release. Keep PR Draft, and do not modify H24/H25/H26 evidence branches or any other repository.

**Disposition:** H28 source/CPU tooling ready for review, but **NO-GO** on inference-quality, model adoption, deployment or merging without remaining gates.
