# H14 combined review: repair delta 1

Scope: delta from the reviewed H14 snapshot only: atomic result/image-freeze persistence, preprocessing timing barrier, added persistence regression test and prospective manifest dependency extension. No source changes, weights, model inference, GPU allocation, commit or push by this judge.

Trusted new manifest: `0d81cc61ecaafb79b52b6a0368f76d550c92f5e4c5ccb05ef468e92a83c5b8d6`. Independently verified all ten manifest members. Frozen task SHA-256 remains `0efd21686fb1fc6255ffe463256f290c8d1cbb2fbcd61ad8d982ae1c5990e6c5`; this delta does not alter tasks or protocol.

Changed runner SHA-256: `b7797e938c9ecb74e71a1ecbd834cd12bc41a70b61539e9299ec68eb59c5b014`; freeze tool: `ae86392c1357011c0a83331d8562f39190c0e1a7367045c504a53aff845351ba`; regression test: `880299165d322165a412002af252bfd2a740cbc9acf6f7a1a98cedb24e77f65f`.

## Findings

- **Must-fix: none in the repair delta.** The prior P2 evidence-retention finding is resolved at `evaluation/heldout_runner.py:48,74`: both destinations now use the existing `training.checkpoints.atomic_json` helper. `training/checkpoints.py:17-26` writes a unique same-directory file, flushes/fsyncs it, then uses atomic replacement and directory fsync on Linux. Failure before publication preserves the previous destination.
- **Worth considering: no new item.** `evaluation/heldout_runner.py:141-142` now synchronizes CUDA before recording preprocessing elapsed time, resolving the previous timing concern. The prior optional semantic-annotation test suggestion remains optional and unchanged.
- **Skip-it / parsimony:** reuse of the existing small helper is appropriate; no new persistence framework is needed. Failed publication can leave an unreferenced temporary file, but cannot substitute it for the last published evidence and does not require another repair for this bounded experiment.
- **Product/guard:** the prospective freeze includes `training/checkpoints.py` and its `training/artifacts.py` dependency. Importing the repaired runner did not import `torch` or `transformers` in an independent CPU check. Remote guard and all-images-before-load sequence remain intact; a new output directory and trusted manifest are still required.

## Independent verification and verdict

Executed the affected real CPU suite: `python -m unittest discover -s tests -p test_heldout*.py -v`: **15 tests, OK**, 0.261 seconds. This includes the actual fault test injecting `os.replace` failure for both `result.json` and `image-freeze.json`, proving the prior valid observations remain readable, plus preparation guard/freeze/parser/renderer checks. The parent's reported RED history was not independently rerun; GREEN was independently executed.

Verdict: **PASS for the reviewed repair delta and closure of this judge's must-fix.** The implementation is suitable to proceed to the authorized prospective remote experiment once other parent-owned gates permit it. This is CPU/source evidence, not successful remote inference, Arabic rendering qualification, security approval or model qualification.

panel: delta - correctness/parsimony/product: fresh-context combined Codex judge (inherited session model); separate security gate outside this report; no d1 substitution.
