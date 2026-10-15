# H3/H7 independent combined review

Verdict: **must-fix**. Reviewed immutable diff SHA-256 `cf8635c7270ac5d103a7cee0f4af627bc04e30d13fb187d9db2331f7f04d9f84`. Evaluation, data, and test sources matched its added content exactly at review time. No repository source was edited.

## Findings

| Priority | Classification | Location | Trigger and observed result | Concrete fix |
|---|---|---|---|---|
| P1 | must-fix | `data/factory.py:159-162,223-226`; `data/export.py:11` | Relabel genuine heldout `gui-toggle-00-487325` to `workflow=save`, `family_id=workflow-save`, `split=train`. Validation and export accept 117 records, including its original heldout ID, toggle goal and state/effects. The allocation check trusts mutable workflow metadata, so it does not prevent test-to-train leakage. | Bind exported records to trusted canonical generated source identities/content and their original allocation; reject changed workflow/family/split, unknown identities, and source content drift. Add this exact heldout relabel probe as a regression. |
| P2 | must-fix | `evaluation/engine.py:151,155,175` | A provider retaining the caller's case changes its `desired_state` to `initial_state` and emits `done_proposed`. The environment stays `saved=false`, while the original desired label was `saved=true`; the runner reports `verified_success=true`, `false_done=false`. The environment snapshots the case but the checker reads the mutable caller object. This is an in-process adapter/reference defect, not a JSON model-output exploit. | Snapshot the task at runner entry and use that detached original for checker labels, budget, and result metadata. Keep provider request copies separate. Add a provider retaining and mutating the caller object to the regression suite. |
| P2 | must-fix | `data/factory.py:170,196-219` | Set a GUI form fixture's `max_steps=1`. Manifest validation accepts it, although it has two annotated actions and the oracle fails final-state verification after its single admitted step. Replay currently ignores the admitted execution budget. | Require the annotated reachable path to fit the declared budget, or replay with the budget enforced. Test a short budget on a form/recovery fixture. |
| P2 | must-fix | `data/factory.py:166-169` | Append a clone with a new task ID and one trailing space in `goal`; initial/desired state and actions are identical. Validation accepts the extra task because the semantic fingerprint hashes literal text. | Normalize incidental goal whitespace for deduplication and retain structural state/action semantics. Add leading/trailing/repeated whitespace duplicate tests. |

## Bars and clean evidence

- **Correctness: FAIL** on mutable expected labels, budget reachability, and duplicate admission above. Existing strict proposal binding, current-generation revalidation, DPI checks, one-action execution, role checks, hidden confirmation targets, and transient recovery remain sound for canonical fixtures.
- **Parsimony: PASS.** Small stdlib state machinery and optional Pillow serve the scoped infrastructure work; no speculative provider framework or paid/model runtime was introduced. The proposed fixes can remain local boundary checks.
- **Product: FAIL** on the explicit leakage/duplication and bounded-label acceptance criteria. The canonical deliverables do meet 60 intent/120 GUI, bilingual balance, DPI variation, 12 workflow families, and the intended independent execution/checker design in ordinary operation.
- **Data/training safety: FAIL** on the relabel bypass. Unknown/benchmark/noncommercial metadata and ordinary same-family split edits are rejected; canonical train export contains 116 exact source fixtures, eight families, and zero heldout task/family overlap. Metadata checks cannot prove rights for arbitrarily relabeled external content, as documented.

## Validation and limits

- Bundled Python via RTK: `python -m unittest tests.test_evaluation -v`: **12/12 pass**; `python -m unittest discover -s tests -v`: **66/66 pass**. Plain discovery initially found zero tests; the explicit test-directory run is the substantive complete-suite evidence.
- Independently executed all six providers over all 180 canonical tasks: every per-task result exactly matches its saved report. Oracle 180 verified; mock 57; wrong-target/stale/invalid/false-done zero. Wrong-target 180, stale/invalid rejections 180 each, and false-done claims 180 for each negative control agree with saved metrics. All seven saved report metrics recompute exactly; their hashes match `evaluation/results/sha256.json`.
- Source task SHA-256: `bdde4ea5326b391172435e29cee77701135c8bfa73805856ffa5dab538bbd35e`; train/dev/test counts 116/32/32 across 8/2/2 families. Frozen bytes reproduce in the suite. Adversarial probes are reproducible in sibling scratch `work/review_evaluation_probes.py`.
- Oracle labels are privileged reachability infrastructure. Heldout 32 tasks represent two designed dependent families. PNGs are not inference inputs; Arabic shaping remains unqualified; drag/drop remains semantic pick/drop. No foundation-model quality, real GUI, physical drag, training run, cloud job, weights, or deployment was reviewed or qualified.

panel: full phase review, fresh combined session-model judge — correctness/data safety FAIL · parsimony PASS · product FAIL · separate security-provider arbitration belongs to caller. This report supplies the inline bars and executable evidence; d1 was not used as a correctness or security gate.
