# Hala ↔ Orcel integration — current product identity and contract gaps

**Status:** compatibility analysis only; integration NOT IMPLEMENTED or qualified. Owner: TheHalfMoon/Hala. Last reviewed: 2026-10-17.

## Canonical target

Orcel is the sole current product identity at [TheHalfMoon/Orcel](https://github.com/TheHalfMoon/Orcel). Its canonical naming and public names are `orcel`, `orceld`, `@orcel/mcp`, and `ORCEL_`. The Orcel `main` identity reference inspected here is `9e64687994b6a3920e541210e5bace116fb28432`. See `docs/canonical/CURRENT.md` in that repository.

HalaMind, HalaType and HalaReliance remain Hala internals. An Orcel adapter is a future integration *boundary*, not a model authority. Hala shall never grant approvals, mint device handles or trusted receipts, lower a risk classification, or directly execute host operations. Only Orcel's existing local authority `orceld` may evaluate and authorize a proposed action, subject to Orcel's actual implemented policy and verified tool vocabulary.

All references to **Aladdin** in older immutable experiment records, historical Git commits and the old `docs/ALADDIN_INTEGRATION.md` are *historical labels and prior reference contracts*, not the current target. Preserve this provenance instead of silently rewriting old evidence or rebranding external material. This document governs the *current* Hala→Orcel target while the old document remains historical for traceability.

## Contract register (proposal, not implementation)

| Surface | Hala development implementation | Orcel published reference | Integration disposition |
| --- | --- | --- | --- |
| Request schema | `hala.request.v1` | `orcel.hala.request.v1` in `docs/orcel/MODEL_INTEGRATION_CONTRACTS.md` | Different versions: requires an explicit, reviewed edge adapter; do not relabel unvalidated payloads |
| Proposal schema | `hala.proposal.v1` | `orcel.hala.proposal.v1` | No public production adapter qualified |
| Task binding | Hala requires `task.task_id`, `step_index`, `max_steps` | Orcel's sample request includes goal/step limits but omits task ID | Must derive and bind authenticated task identity from Orcel authority; never accept model-supplied identity |
| Session binding | Hala requires `device.session_id`, tenant and observation bindings | Orcel's sample includes device ID/OS/locale and generation but omits session ID | Must bind from trusted Orcel session/observation state, not infer from display/UI text |
| Device action vocabulary | Typed Hala proposals checked against per-request allowed actions and ceiling | Orcel's draft includes `semantic_invoke`, `click`, `type`, etc. | Map each proposal only to an actually qualified Orcel action, with strict ceiling and fresh reference checks |
| Completion | Hala outputs untrusted proposals, not success receipts | Orcel requires independently verified outcomes | Orcel alone checks postconditions; no AI self-asserted `done` is a receipt |
| Model lineup | Research baseline Qwen3-VL-2B; retained shared LoRA has narrow TEST evidence; HalaReliance deterministic by default | Historical adopted Orcel proposal assumes Holo4-35B-A3B main, Liquid d1 optional | The named checkpoint is not an implemented endpoint. Select model only after shared representative evaluation, revision/license/VRAM/cost qualification and founder decision |
| Authentication and serving | Development bearer-bound loopback mock; real service not release-qualified | Orcel planning requires founder-side inference, local device execution, bounded authenticated proposals | Keep independent server-client auth, revision pins, limits, cancellation, privacy and separate release gates |

## Read-only source references

- Orcel: `docs/canonical/CURRENT.md`, `docs/orcel/ORCEL_AI_SERVER_ARCHITECTURE.md`, `docs/orcel/MODEL_INTEGRATION_CONTRACTS.md`, `docs/orcel/EXECUTION_FRONTIER.md`, `crates/orcel-contracts/src/lib.rs`.
- Hala: `hala/contracts/`, `hala/serving/`, `docs/ALADDIN_INTEGRATION.md` (historical), `docs/CURRENT_STATUS.md`, `docs/ROADMAP.md`.
- The Orcel design documents are historical/planning sources; they do not override currently implemented local authority contracts or mark AI integration complete.

## Explicit future admission gates

1. Independently verify current Orcel service/kernel contracts and target API with immutable source revision; resolve model-request fields and auth boundaries.
2. Create a versioned Hala↔Orcel conversion interface in Hala only, tested with complete request/response examples, both languages, stale-frame/session/tenant rejection, permission ceilings, cancellations, explicit abstention and unsupported actions.
3. Qualify Hala real inference with representative held-out closed-loop tasks, correct Arabic/RTL behavior and measured VRAM/latency/cost on eligible compute.
4. Implement and evaluate an Orcel-side adapter **only in Orcel's own independently authorized engineering scope**, not by modifying its repo from Hala work.
5. Separately qualify security and policy, real Windows frame/DPI/device observations, approvals, native actions, outcomes and production ops. A test in a synthetic executor does not pass the native Orcel gate.

**Current finding:** documentation can name the correct product, but Hala is not natively integrated with Orcel and no real-model precision candidate is adopted. Do not declare production-ready.
