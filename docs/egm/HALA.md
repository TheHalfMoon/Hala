# EGM in Hala — architecture review, not model inference

Hala uses the upstream [Google Elephant–Goldfish Model](https://github.com/google/elephant-goldfish-model) **as a design-review workflow**, not as a new model or an alternative to Qwen. The current design work is [Hala RTL visual-action study](../designs/hala-rtl-visual-action-learning-egm.md). It is a **proposal, unapproved and unreviewed**.

## Source/version

- Upstream Google repository: `google/elephant-goldfish-model`.
- Reviewed upstream main Git commit: `403d8881d62022490637d740657b33f01e321ea8` (2026-10-30).
- License: Apache-2.0, with upstream original copyright and attribution retained in upstream. No upstream files are copied/vendored into Hala. The plan text is original Hala work.
- Requirements used: `resources/document_standards/feature-spec-standard.md`, `resources/document_standards/implementation-readiness-standard.md`; isolation guidance `docs/codex.md` and `skills/egm-review/SKILL.md`. Do not assume upstream `main` is unchanged; verify the chosen pinned release/commit before invoking.

## Install EGM in a supported coding-agent harness

Official upstream Codex instructions, after owner review of the code and current installer:

```sh
codex plugin marketplace add google/elephant-goldfish-model
codex plugin add egm@egm
```

This installs the **whole** EGM plugin, not only `skills/`: reviewer personas and `resources/document_standards/` are required. A fresh agent session is necessary. Do not silently install globally on an owner workstation; use an owner-authorized isolated environment, clean temporary caches, and observe Hala's zero-cost restrictions. **This H330 PR does not perform a plugin installation or model-based EGM review.**

In the agent workspace use the Hala repository at the proposed Draft HEAD and select the installed `egm-review` skill for `docs/designs/hala-rtl-visual-action-learning-egm.md`. Optionally specify `EGM artifact directory: docs/egm/reviews` as a project-relative directory so accepted text reports can be stored in Git on a new Draft, never in private Kaggle or model dataset storage.

## Hala-specific Goldfish packet

Start a **fresh isolated, read-only** reviewer with only:

- Target: `docs/designs/hala-rtl-visual-action-learning-egm.md`
- Standards: upstream pinned `resources/document_standards/feature-spec-standard.md` and `implementation-readiness-standard.md`
- Reviewers: pinned upstream `agents/zoom_out.md`, `agents/comprehension.md`, `agents/design_critic.md`, `agents/implementation_readiness.md`; use native EGM `egm-review` workflow, not a homemade simulation
- Hala source-only reading list: `AGENTS.md`, `h199/native_action_sft.py`, `h319/beaconboard_dynamic_research.py`, `h325/two_step_train_candidates.py`, `h326/rtl_relative_geometry_audit.py`, `h328/original_family_visual_split.py`, `h329/railkit_source_provenance.py`.
- Context boundary: NO parent conversation, memory, private H324 receipts, raw images/screenshots, gold action/points, model predictions, hidden H173 OOD, quarantined H250/H251 assets, or H293/H294 blocked private retrieval.

Reviewer outputs: findings grouped by **Blocker / Major / Nit**, with source path/section and evidence, alternatives, failed assumptions, potential counterexamples, test needs and unresolved questions. Empty findings alone do **not** qualify; isolation and actual source/standards reads must be trace-verifiable. If fresh-context tools are missing, or the reviewer did not read source or standards, record **REVIEW_BLOCKED**, not PASS. Parent assertions, prompt summaries and a normal same-thread assistant critique are not Goldfish review evidence.

Once reviewed, independently resolve design issues and document decisions in a new Draft. Design agreement alone cannot grant H199 data admission, legal media rights, permission to use paid compute, open barred OOD, deploy/release Hala, or authorize publishing private model data. Owner approval and all other scientific gates remain.

## Why this changes engineering

Without EGM, repeated success of mocked/source-owned browser CI risks being mistaken for improved model weights. The EGM design demands alternatives and falsifiable acceptance conditions before allocating GPU time: a source-disjoint never-exposed evaluation, lawful TRAIN admission, paired BASE/ADAPTED model-time action results and uncertainty. This reduces speculation and protects the existing source/evaluation boundaries; **it does not itself make Hala more accurate**.

A separate study may implement training only after review/rights/gold firewall clearance. No EGM component should be imported in the policy inference path, training dataset, action executor, or CLI runtime. No new dependency, network call, API key, media upload, or data access is introduced by this PR.
