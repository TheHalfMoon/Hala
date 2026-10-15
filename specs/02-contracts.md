# H2 Component contracts

## Goal
Implement bounded HalaMind, HalaType and HalaReliance interfaces and mock inference without authority or native computer execution.

## Coordination
Depends on: H0
Surface: hala/contracts, hala/mind, hala/type, hala/reliance, tests/test_contracts.py
Review: full

## Requirements
Pin reference Aladdin main 0c3c86c8acfb57e3dd3ce83c83ee8cfd3b5035c1. Its MODEL_INTEGRATION_CONTRACTS.md is adopted planning direction with proposed wire schemas. Do not claim kernel capability qualification. Preserve request/task/device/session binding, observation generation, action vocabulary, method ceiling and model/prompt/config revision. Reject unknown fields, unsupported actions, stale references, invalid coordinates, forged authority and success claims. Proposal validation is not authorization. Build a deterministic mock with explicit metadata.

## Acceptance criteria
- Strict immutable records validate bounded requests and responses; unknown nested fields and nonfinite numeric values fail.
- Actions reference observed elements and allowed vocabulary and obey method ceiling and frame bounds.
- Confidence and abstention are bounded; done_proposed only recommends local verification.
- Tests cover bindings, malformed input, stale observation, unsupported tool, privilege attempts, malicious text, cancellation and false completion.
- Interface exports the three official components and runs offline without weights.

## Hardstop
Do not invent executable tools, receipts, permissions, or real model performance. No Aladdin writes.

## Out of scope
Production serving, training and real inference.
