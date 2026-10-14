# H9 routing and executable cost research

## Goal
Measure deterministic routing overhead and expose transparent cost scenarios without claiming model performance or adopting prices.

## Coordination
Depends on: H2
Surface: hala/reliance, experiments/cost, tests/test_cost.py

## Acceptance criteria
- Routing only recommends deterministic/mind/type/observe/ask/abstain/escalate/budget exhaustion; cannot grant authority.
- Cost calculator validates bounded finite inputs and models GPU/idle/storage/network/retry/success/failure and warm/cold work.
- Scenario table covers 10/50/100/500/1000 users and $5/$8/$10/$15/$20 monthly price research.
- Synthetic assumptions are labeled; measured records required for actual operating-cost qualification.

## Hardstop
No adopted pricing or proprietary runtime API requirement. No synthetic zero-cost inference claim.
