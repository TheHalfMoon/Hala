# Foundation repair delta review 2

**Verdict: C1–C4 resolved for the implemented source/helper scope; no remaining must-fix found in these repairs.** This is not model, processor, LoRA or release qualification.

Immutable evidence chain:

- Original review: `foundation-review.diff`, SHA-256 `8251045a4f84c22310e46e1e5069872292476d1db5eec81a77e86cb9395ca17d`; findings in `foundation-combined.md`.
- Training repair review: `training-final-review.diff`, SHA-256 `d5d9d225a044ca9c0b278b76738fc168fb8358f8b45d9b068fe0eeeb9deeeb12`; evidence in `training-delta-1.md`.
- Final cost-only delta: `cost-final-delta.diff`, SHA-256 `de948a5cbbffc3af14ee7bbf8a4a3522dd588f8e796aa1e7aa165ac88fcaa69e`.

| Original finding | Final status | Verified result |
|---|---|---|
| C1 media provenance/integrity | Resolved | Exact JSONL plus every normalized local PNG path/digest/family/original provenance are audited before runtime/model loading. All 58 English candidate records/media validate. Independent mutation of the last image rejects before a runtime probe; completed resume rejects changed media as well. |
| C2 restart/checkpoint loss | Resolved | Complete immutable versions bind adapter, optimizer/scaler/RNG, identities, all saved metrics and update evidence through hashes; one atomic pointer publishes the pair. An independently injected pointer-replace interruption preserves the previous complete version. Completed resume finalizes saved metrics without a probe, new torch/peft/transformers import or model allocation. Partial resume restoration is inspected source plus helper tests, not measured numerical equivalence. |
| C3 nonfinite cost output | Resolved in cycle 2 | `experiments/cost/calculator.py:67–72` now rejects nonfinite derived values and per-verified cost before returning. Independent probes reject `success_rate=5e-324`, `month_hours=5e-324`, and an overflowing `month_hours=1e-308` scenario. Large magnitudes/huge integers reject; finite small denominators, zero-work/zero-price cases, maximum bounded inputs and the default 25-row table serialize with `allow_nan=False`. |
| C4 malformed diagnostic JSON | Resolved | Strict raw-answer parser rejects duplicate/nested duplicate keys, nonfinite constants and exponent overflow before exact integer-point validation; valid point output still passes. |

Testing: 83/83 full CPU tests passed for the training-repair snapshot. After the isolated cost fix, the independent cost reviewer passed 4/4 cost tests plus the boundary probes above. Applying that final delta in memory to the prior immutable snapshot exactly reproduces current calculator/test contents. The training paths were unchanged and were not re-reviewed in cycle 2. No source edits during either independent review; no account-limit blocker prevented this review.

Final cost source SHA-256: `bced9d0548ef8c9fb5a5656aab05340c7ad9049f72ead638753e6d86ad2528ee`.
Final cost-test SHA-256: `0171a7f790f817c1633d679b4316a8aeaadfae091de985bc21ed07fcc4d9ac77`.

Parsimony/product: the added artifact, mask and checkpoint helpers serve concrete integrity boundaries; candidate terminology and unqualified flags are accurate. The converter prepares English initial-state logical-action SFT inputs, excludes unqualified Arabic image shaping and holds out dev/test families. The tested language-only LoRA regex is not a reported regex bug and does not establish actual model-module qualification.

Qualification limits persist: no foundation weights locally, no remote job, no actual processor invocation, no LoRA step, no real adapter/optimizer checkpoint execution, no model-fit or adapter-quality evidence, no continuous/resumed numerical comparison and no confidence calibration claim. CPU checkpoint probes use synthetic bytes. These source repairs do not replace the parent's separate security review or the future real-runtime qualification gates.

panel: delta cycle 2 — correctness ✓ independent source/tests/mutations · parsimony ✓ combined inline bar · product ✓ candidate status · security — prior separately coordinated review; no actual processor/LoRA/model qualification
