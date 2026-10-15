# 0001. Start with independent CPU qualification

Status: accepted (2026-10-14)

## Context
Hala begins in an empty repository, with no qualified remote GPU. Aladdin is developed separately and retains all computer authority. Early work must remain inexpensive and reproducible.

## Decision
Use Python 3.11+ standard-library contracts, deterministic synthetic fixtures and a visibly mock service before introducing model-runtime dependencies. Use one shared Mind/Type checkpoint as the provisional research direction; actual model selection remains unqualified until measurements. Keep Reliance optional and advisory. Pin and inspect model metadata without downloading weights locally.

## Consequences
CPU CI can validate protocol and state transitions now. It cannot establish multimodal quality, latency, cost, Arabic comprehension or model safety. Optional remote dependencies and adapter training require runtime and license qualification, unchanged-base evidence and genuine gradient measurements.
