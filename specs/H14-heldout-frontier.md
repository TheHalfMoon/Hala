# H14: frozen coordinate protocol and prospective bilingual held-out GUI evaluation

Status: in progress. Depends on: H5 diagnostic and H3 infrastructure built in this run. Surfaces: new experimental protocol/heldout modules, frozen evaluation specifications, remote runner and dedicated tests/docs. Production contracts remain separate.

Use one shared pinned Qwen3-VL-2B unchanged model first. Freeze an explicit normalized 0..1000 protocol, half-open target rectangles, image-pixel/DPI assumptions, deterministic mapping and failure behavior. Reject malformed/nonfinite/bool/extra/duplicate coordinate outputs. No execution authority or completion receipt from the model.

Generate 64 prospective GUI task instances, balanced English/Arabic, across eight original families unlike the prior four-button diagnostic and the existing training families. Vary layouts/dimensions/targets/distractors and task semantics; disclose family dependence. No training or test-set prompt tuning. Freeze specifications and protocol hashes before evaluating; remote-render Arabic only with actual RAQM and a declared font, freeze resulting images before inference. Keep expected targets/private state out of model inputs. Simulated executable environments may report complete-task success only for their declared fixture scope, never native Windows/browser task qualification.

Acceptance criteria:
- [ ] Strict protocol specification and deterministic failure/edge/DPI tests.
- [ ] Frozen source manifest with 64 instances, eight independent families, original rights and balanced languages; no old calibration task/image reuse.
- [ ] Remote runner verifies hashes, renders all inputs before loading model, records actual provenance/latency/allocated and device residency measures, schema validity, hit/wrong-target/miss rates and English/Arabic intervals.
- [ ] Negative providers demonstrate scorer rejects wrong targets, invalid outputs and false completion; no model sees expected target/terminal state.
- [ ] Relevant tests and full CPU suite; genuine Jev/Graft and independent PStack delta review.
- [ ] Actual free-GPU evaluation when available, source/dataset frozen before inference; no fabricated score on allocation failure.

Assumptions: eight authored families are a narrow reproducible GUI fixture sample, not a population-representative desktop benchmark. Wilson intervals are descriptive; provide family-level summaries and clearly label correlation. GPU allocation is external; CPU implementation/reviews proceed regardless. No paid activation or local weights.
