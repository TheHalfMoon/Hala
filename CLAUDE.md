# Hala implementation conventions

Read PRODUCT.md, docs/MASTER_PLAN.md, docs/ROADMAP.md and the active spec. Python 3.11+; begin with standard-library contracts and unittest so CPU-only CI can run without model dependencies. Optional training/runtime dependencies require exact versions and qualification. All public claims distinguish implemented, tested and qualified. No large weights locally. No modifications to Aladdin. No customer secrets or telemetry. Prefix shell commands with rtk.

Use signed DCO commits; no shared-history rewriting. Feature branches and review evidence per grain; founder controls merge and production deployment. Arabic strings are test data; code and prose stay English.

## Capabilities

- correctness, parsimony, product: PStack ps-review with fresh-context Codex judges; available.
- security: separate fresh-context judge for triggered surfaces; d1 is advisory only.
- docs lookup: GitHub CLI, official metadata HTTP, Exa; available.
- browser: unified computer use available; live native computer control unavailable.
- gate tools: bundled Python, unittest, installed Jev 2026.919.0, Graft 0.21.1. Probe actual help; installed Jev differs from skill examples.
- canvas: absent; founder's detailed autonomous directive supplies planning choices.
- subagents: available for fresh-context PStack builders and judges, as required by installed skills.
- model-tiers: use inherited session model and reasoning effort.
- GPU: existing free Colab T4 has actual FP16 tensor and unchanged-model diagnostic evidence; held-out task, training and quantization qualification remain pending. Local d1 CPU runtime is online, but is not a qualified Hala foundation.

Graft telemetry is disabled. Its installed build/check/blast commands qualify source graph freshness and impact; it is not an isolated execution sandbox or correctness verifier. Jev makes typed judgments and cannot produce prose review findings; pair it with independent PStack review.
