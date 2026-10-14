# H3/H7 independent delta review — cycle 1

Verdict: **PASS for the reviewed delta; all four original must-fixes closed**. No new actionable findings.

Immutable delta SHA-256: `175de9ed185dec643e62dcfac28561bbca4f5d4ddfd7ae7d39b62590155be7d6`. The four changed files' live Git blob IDs match the delta exactly: factory `ef7fd86`, documentation `c594be3`, engine `610efa8`, evaluation tests `93f43d7`. Original review artifact remains `evaluation-combined.md`; its failures describe the superseded source, not this repaired delta. No repository source edits were made by this reviewer.

| Original finding | Independent adversarial evidence after fix | Status |
|---|---|---|
| Heldout content relabeled into training | Metadata-only relabel rejected. A heldout task rewritten to a valid train ID/document/workflow/family/split passes structural validation but export rejects its full-record digest. Changed goals, added oracle steps, and seed relabeling also fail source binding. Exact seed-7 generated records remain reproducible and export 116 train records. | CLOSED |
| Mutable caller expected state/budget/identity | Provider overwrites desired state with initial state plus task/family/split/language/suite metadata; runner still records original identity and returns `verified_success=false`, `false_done=true`, `saved=false`. Providers changing admitted budget 3 to either 0 or 60 both execute exactly the original three steps. | CLOSED |
| Annotation replay ignores admitted budget | GUI form budget 1 and recovery budget 2 both reject with `Annotated path exceeds admitted task budget`. These include the transient attempt/observation costs. | CLOSED |
| Whitespace semantic duplicate admission | Trailing whitespace, NBSP, narrow NBSP, and newline/tab duplicates reject as `Duplicate semantic task`. The added suite separately passes fullwidth compatibility and Arabic canonical-composition duplicate probes. | CLOSED |

Fix locations: `data/factory.py:34` normalizes goals; `data/factory.py:174-183` binds fingerprint and path budget; `data/factory.py:184-188` binds task/document identity; `data/factory.py:240-256` regenerates exact original source and binds full-record hashes; `evaluation/engine.py:153` snapshots the caller fixture before provider execution. No generic external-dataset importer was added.

Validation: bundled Python through RTK, `python -m unittest tests.test_evaluation -v`: **16/16 pass**. Independent scratch `work/review_evaluation_delta_probes.py` and `work/review-evaluation-delta-probe-results.json` contain the additional probe reproduction and compact outcomes. The exact canonical training export remains byte-identical: 116 records, eight families, SHA-256 `328cf96c8b43abe2e280c1e68824a26957e5d3d140cbc9a9945e059092247908`. Frozen tasks remain SHA-256 `bdde4ea5326b391172435e29cee77701135c8bfa73805856ffa5dab538bbd35e`. Independently rerun heldout oracle: **32/32 final states across two families**.

Bars: **correctness PASS**, **parsimony PASS**, **product PASS for H3/H7 scoped infrastructure**, **data/training safety PASS for exact original-source importer**. The source regeneration and detached task snapshot are local repairs with clear boundaries; tests cover the demonstrated failures rather than treating advisory model scores as gates. No d1/Jev score was used to establish this verdict.

Limits unchanged: oracle labels remain privileged infrastructure reachability; task variants are dependent; PNGs are not inference inputs; contextual Arabic shaping and physical drag/drop remain unqualified. No real-model quality, cloud/training run, weights, external rights-verification service, real application executor, or deployment is qualified. Changes to other training files and the root full-suite gate are outside this delta review.

panel: fresh combined delta judge (session model) — correctness PASS · parsimony PASS · product PASS · data/training safety PASS · separate security arbitration belongs to caller.
