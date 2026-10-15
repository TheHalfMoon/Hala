# H19 — Compatible bilingual development tasks

Status: source implemented and CPU-tested; actual media/model evaluation pending. Depends on: verified foundation 0638137724eb9a5bde1e6133aef976020e11fae3. Review: full.

## Scope and coordination

New isolated model-development branch; PR1 remains unchanged. Own new `development/` package, `tests/test_development.py`, `tools/build_development_data.py`, `docs/H19-bilingual-development.md`, and `experiments/development/` only. Do not edit frozen H14/H17 files or H18 historical audit. Other branches may own Reliance diagnostics or review records.

## Requirements

Author a small CC0 synthetic dataset with genuinely bilingual English/Arabic user goals, observation content and completions. Use one shared checkpoint, separate task prompts for Mind, Type and advisory Reliance. Mind examples cover plans, tool selection, recovery, clarification and abstention; Type covers strict normalized coordinate click/type/drag proposals. No executable code or model authority. Use explicit independently separated application families (at least four train, two dev, two test); test application families must not appear in train/dev and must not be frozen H14 families. No private target identity, expected box, answer or final state in model input. Do not hide dataset ambiguity through opaque IDs in goals.

Implement bounded synthetic multi-step task state transitions with a separately implemented final-state verifier. Include action errors, recovery opportunities, unnecessary calls and unsafe proposals in trace scoring. Goal success must depend on actual final state, not proposed actions. Prompts and labels must be versioned; strict parsers reject extra fields, duplicates, nonfinite/boolean/out-of-range coordinates and unknown actions. Model output cannot execute arbitrary code or grant permission. Keep native integration explicitly unqualified.

Arabic media generation must fail closed unless Pillow RAQM and pinned font shaping are available. Local tests may qualify text/state logic without claiming local Arabic media quality. Generate actual qualified media only on verified Linux remote; archive font/version/dimensions/hash provenance. Freeze independent evaluation trajectories and prompt/settings before real baseline output is observed. Include deterministic label/postcondition validation and adversarial negative tests, split leakage checks and independent expected state verification. Do not train or download models in this source phase.

Produce a clean minimal source interface for remote unchanged-base evaluation and subsequent use of the qualified LoRA mechanism. Training adaptation belongs in a later reviewed phase. Keep data small (roughly 80–160 training turns, 12–24 dev/test episodes total), clearly disclose templated/dependent examples and limited coverage. Quality matters more than count.

## Acceptance

- [x] New bilingual task rows, license/provenance and disjoint family split manifest generated deterministically.
- [x] Semantic Mind, normalized Type and conservative Reliance labels validated; real multi-step oracle episodes complete under independent verifier.
- [x] Invalid/unsafe proposals fail closed; adversarial state and label corruption tests meaningful.
- [x] Strict bounded model prompt/response/trace interface and reproducible frozen baseline inputs ready.
- [x] RAQM media qualification gate and hash capture implemented without pretending unexecuted media was tested.
- [x] Scoped red/green tests, integrated CPU suite, fresh PStack combined/integrity review, genuine Jev and Graft evidence recorded before signed DCO commit.

No paid compute, large local weights, Aladdin edits, foundation PR expansion, release claims or founder merge. Assumed internal choices: new experimental protocol may use explicit semantic tools for Mind and normalized visual actions for Type, with deterministic policy checks and simulated executor.
