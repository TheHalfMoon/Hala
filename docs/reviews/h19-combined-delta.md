# H19 independent combined repair delta

**PASS for this delta.** The prior recovery-state must-fix is resolved. Requested-kind enforcement and independent Mind semantic accounting address the prior evaluation-quality concern without supplying corrected answers to Type. No new must-fix found within these repairs; the original broader review and its data/runtime limitations remain applicable.

Reviewed current `work/Hala-bilingual` simulator, runner, verifier, new regressions and documentation. No tracked source edits, commits, models or GPU execution by this reviewer.

## Independent verification

- Replayed the original premature-observation reproduction through the actual simulator. After pre-observation, correct edit/drag and first failing save, immediate retry now produces **success=false, action_errors=1, environment_errors=1, recoveries=0**. The new failure explicitly clears `observed`. A subsequent real `observe()` followed by save then succeeds with recovered=true. The external probe now asserts both rejection and successful later recovery.
- Replayed the original wrong-kind/wrong-subgoal callback through the actual runner with mocked rendering. First requested plan receives `next`: recorded requested_kind_correct=false, consumes one model call, increments model_errors and does not invoke Type. Later wrong semantic subgoals can still select a visual tool; Type sees the original full user goal. Final task success is true while **mind_calls=6, mind_correct=2, mind_errors=4, mind_accuracy=1/3, plan_calls=1, plan_correct=0, plan_accuracy=0, model_calls=10, model_errors=1**. This is the intended separation of persisted-state success and Mind semantic quality, not a false all-correct score.
- Asserted Type prompt JSON remains exactly `goal`, `observation`, `request`; it has no oracle-corrected subgoal or expected-action field. Source inspection confirms `verify_mind_semantics` receives only public goal, visible observation, response and requested kind; it imports neither oracle nor simulator. It extracts the public quoted English/Arabic requested value/destination and compares the narrowly authored plan/next-step semantics. The final-state verifier remains independent of model success claims.
- Independently ran **16 targeted tests, all passed in 0.507 seconds**, including bilingual public-state semantic checks and both new regressions. No full-suite rerun; the integrated result remains builder/root evidence.

Reproducible probe: `work/h19_combined_probes.py`, run from task workspace with `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h19_combined_probes.py`. Initial probes were retained and strengthened with assertions rather than replaced by a new implementation. A shell `Get-FileHash` attempt was unavailable in this environment; source hashes were then computed directly by Python hashlib in the same probe.

## Bars and limits

- **Correctness:** recovery freshness and requested-kind behavior now match requirements; semantic errors are explicitly counted.
- **Product/data:** task final success and Mind decomposition accuracy remain separate visible metrics. The narrow public-goal parser is appropriate to these authored quote-formatted tasks; it is not a generic language-understanding oracle or native task verifier. Shared family workflow templates and sparse Reliance labels remain disclosed limitations from the original review.
- **Parsimony:** one recovery reset and a small separately implemented semantic checker are proportionate. No oracle feedback channel, native executor, generic planning framework or authority mechanism was introduced.
- **Evidence/freeze:** code changed before any model outputs were observed; any previous source freeze must be regenerated and independently archived. Arabic rendering, real model inference, durable artifacts and useful quality remain unqualified. No result here substitutes for separate integrity/security review, Jev/Graft, CI or founder authorization.

Exact reviewed SHA256 values:

```text
development/runner.py c2f491c731ff2656de9bfe8e3b7bbf15eeca677f41aebafd07514a13e2879fb1
development/simulator.py 461f8e931336cf6a3a28225ed815616034277b66dd266e2fdf768ddda34f4141
development/verifier.py 79cd886a0fc5f9f2022fe7d8b79687de7df53e84a25d2b4977b76e842a7c6ceb
tests/test_development.py 3486a68bedf92ee96b756bcdf75c1f8acf810814441b07f76b4d49566639ebe3
```

Panel: fresh independent combined correctness/product/parsimony delta; original exploit replayed, no source edits. Genuine Jev typed probabilities are advisory; this report is based on source inspection and executed probes, not interpreting probabilities as prose findings.
