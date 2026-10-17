# H19 independent security and data-integrity review

Verdict: fix one P2 simulation-integrity finding before baseline inference. No native execution or model authority path found in the scoped source.

Reviewed new `development/*.py`, `tests/test_development.py`, `tools/build_development_data.py` and `docs/H19-bilingual-development.md` against AGENTS.md, CLAUDE.md, PRODUCT.md and the H19 spec. Base supplied: `0638137724eb9a5bde1e6133aef976020e11fae3`; scoped files are currently untracked. This judge did not author or mutate repository source. No inference, download, GPU work or full-suite rerun occurred.

## P2: pre-failure observation satisfies a later recovery requirement

Location: `development/simulator.py`, first transient-save-failure branch and `observe()`.

`observe()` sets the persistent `observed` flag before any failure. The first failed save sets `failed_once` and `status`, but leaves that flag true. A subsequent save therefore passes the retry guard without inspecting the newly failed state, and is counted as both successful and recovered. This violates the specified requirement to inspect the transient failure before retrying and can inflate recovery performance for a model that makes a redundant observation early.

Reproduction in `h19_integrity_probes.py`: construct an English transient-failure episode; call `sim.observe()` immediately; apply the oracle type and drag; manually click save twice without another observation. The first click produces `transient_failure`; the second produces `verify_final(...) == True`, `recovered == True`, `action_error == False`. No oracle is called between the two clicks.

Minimal fix: clear `observed` when introducing a new transient failure (or record an observation generation tied to the current failure). Add a regression asserting an earlier observation cannot permit retry, then assert a fresh observation does permit it. Keep actual final-state verification separate.

## Other boundary checks

- Exact deterministic dataset validation rejected completion, family-split and provenance substitutions.
- Source-only freeze with an independently pinned manifest hash rejected modified dataset bytes. The review assumes callers independently archive that hash as documented; it does not posit an attacker changing the trusted anchor.
- Fifteen malformed/deep/duplicate-response checks across Mind, Type and Reliance rejected cleanly. An escaped lone surrogate accepted by the first Type parser was rejected by the simulator's second parse before state mutation; no runner crash or false success reproduced.
- Canonical media filenames reject explicit traversal in manifest records; model output does not supply file paths. Pinned image hashes detect byte substitution under an unchanged trusted manifest. Live symlink races and actual Linux RAQM rendering were not exercised locally and are not claimed qualified.
- Prompts contain public goal and visible observations; private expected state/boxes and oracle completions are outside the callback interface. Final verification reads simulator state and persisted fields rather than model receipts. Native qualification remains explicitly false.

Executed probes with the bundled Python via `rtk proxy`; final exit code 0. Reproduction script is outside the repository at `work/h19_integrity_probes.py`.
