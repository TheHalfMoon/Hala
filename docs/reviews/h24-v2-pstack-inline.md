# H24 V2 — PStack-style inline review (NOT independent security signoff)

**Scope:** Hala only; Draft PR #5; `h24/mind_dev.py`, `tools/build_h24_kaggle.py`, `tools/verify_h24_dev.py`, seven new V2 tests and `docs/H24_MIND_DEV_PROTOCOL_V2.md`. 2026-10-18.

**Correctness:** V1 source and raw measured outcome remain pinned and immutable. V2 is explicitly named `fixed_allowed_pairs_v2` and has a distinct Kaggle notebook slug, result schema and auditor selection. Only five normalized `tool/subgoal` pairs are allowed; crossed pairs, free-text, unsupported verbs, unverified model output and altered protocol markers fail closed. Test cases are strictly DEV and the train/TEST families are excluded. Exactly **314 local Python tests passed, one skipped**, including all historic H19/H22 regression/freeze tests.

**Security/authority:** V2 prompts are projected from the already-redacted V1 public DEV observation; no synthetic `protected` labels or expected answers are exposed. Orcel alone owns device execution and authority; Hala yields untrusted suggestions. No model-supplied action is executed during the benchmark. Remote free-T4 Kaggle admission, hash-pinned source/model and create-only saved evidence apply. *Fresh-context independent security/PStack judge has not reviewed the changed head.* Jev typed probabilities are advisory, not security approval.

**Product/science:** V2 changes instruction formatting and allowed subgoal vocabulary on deliberately reused **20 synthetic DEV cases** after observing V1 failures. Any improvement may reflect prompt-format alignment rather than general reasoning or Arabic task proficiency; this is *not a new independent holdout* and never qualifies HalaType, native closed-loop Orcel, production, fine-tuning or a precision. The untouched H24 TEST must not be used for iterative tuning.

**Parsimony:** Versioned protocol branches through the existing CPU scorer, remote runner and evidence verifier, without duplicating a model runner, changing LoRA, or downloading weights to Windows.

**Executed tools:** genuine Graft build/check passed (wiring graph only), genuine Jev 10-file typed advisory passed with exact file hashes retained in `h24-dev-jev.json`, Alibaba Open Code Review not separately executed as an LLM judge. No PStack fresh-context approval.

**Disposition:** allowed to publish to **Draft** Hala PR #5 for further research. GPU DEV V2 may be sampled prospectively within free Kaggle terms and independently audited; no main merge, no Orcel changes, no model/precision adoption, and no paid resources.
