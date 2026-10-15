# H20 independent combined PStack review

Verdict: one P2 malformed-response robustness fix required. Recorded diagnosis and conservative non-adoption conclusion are supported. This reviewer did not author the change, alter repository source, or rerun inference/full suite.

Scope: `tools/diagnose_reliance.py`, `tests/test_reliance_diagnostic.py`, H20 spec/docs and recorded diagnostic/config-repair evidence. Reviewed against AGENTS.md and CLAUDE.md; fixed loopback calls are diagnostic only, with no action executor or permission grant.

## P2: a bounded numeric probability can abort evidence collection

Location: `tools/diagnose_reliance.py:126`, with `run` exception handling at line 199.

`route_answer` calls `math.isfinite(x)` on arbitrary JSON integers before rejecting values outside [0,1]. A syntactically valid response of less than 5 KB can contain `10**4000` as one probability. Python parses that integer successfully but `math.isfinite` raises `OverflowError`, which neither `route_answer` nor the run loop translates into a rejected/error prediction. A malformed local-runtime response therefore aborts the diagnostic and prevents publication of completed evidence, instead of being counted as an error. The current tests cover NaN and missing fields but miss this numeric boundary.

Minimal fix: reject `x < 0 or x > 1` before `math.isfinite(x)` after the existing bool/type guards; arbitrary integer comparisons do not require float conversion. Add a regression using a parsed, bounded JSON integer response and require `ValueError`/an error prediction. No model rerun is needed to verify this change; retain the existing measurement and its executed-source hash.

## Evidence replay and assessment

- Independently reconstructed all 32 recorded test cases, training corpus, learned classifier, each classifier prediction, every direct d1 request body/hash and parsed response. Recomputed every summary count, correctness/error/abstention/unsafe count, median and P95 latency. All match. The four stdio wire hashes also match their bytes.
- Confirmed source hash `7260f83e484ce51d86a7278a93e29c16d16d24276a0c232ffe757279966bd7db` and comparison hash `bc5fde251db206991c7675215f1a75cbd54829022171e8ae18a94527744e6d04`. Classifier serialization is exactly 5,070 bytes.
- Recorded d1 accuracy is 13/32 with seven unsafe route suggestions out of eight unsafe/ambiguous cases; every comparison returned HTTP 200. Deterministic and naive Bayes scores are 32/32 on shared dependent templates. Documentation clearly discloses family separation versus template reuse, unequal trusted-fact inputs, and absence of adoption qualification. No held-out natural-language generalization claim is warranted or made.
- Encoding controls meaningfully separate native UTF-8 transport, escaped JSON, CP1252 corruption and invalid schema. Fresh child UTF-8 repair is distinguished from the existing desktop process environment. Installed source/runtime/model hashes are preserved; no installed-source mutation or server restart is part of this change.
- The config repair record contains selected nonsecret settings, hashes and backup path rather than the complete configuration. Independently checked the backup's original hash and compared parsed current configuration against backup after removing the one UTF-8 setting; no configuration content or credentials were printed by the probe.
- Tests exercise encoding, split leakage, malformed/nonfinite responses, safety precedence, unknown-token abstention and replayed transport evidence. They are appropriate to a diagnostic; perfect synthetic classifier performance is explicitly unqualified.
- Cost claims stay bounded: in-process versus localhost latency differences are disclosed; model file bytes are not presented as RSS/CPU usage/total cost. No paid services, training, dependency or weight downloads are introduced.

Offline replay and adversarial reproduction are in `work/h20_review_probes.py`. This review is independent PStack evidence, not a substitute for the separately required Jev/Graft or founder merge authorization.
