"""H61: complete original corpus, capability-balanced curriculum and real grounding training.

H60 proved the original DragOn TRAIN source, recovered a partial slice of the
slider archive, and measured an honest negative result: a 60 step adapter
learned the JSON output contract and then collapsed onto a constant degenerate
point, so real GUI drag task success stayed at 0 of 48.

H61 attacks the measured deficit on the training side:

* `h61.source_export` reads the pinned original TRAIN archives directly with
  byte-range verification and a whole-archive SHA-256 proof, and materialises
  only the screenshots a capability-balanced curriculum actually needs.
* `h61.curriculum` turns those originals into a reproducible, capability
  balanced curriculum with screenshot-level deduplication.
* `h61.prompt_contract` is the single prompt template shared by the training
  curriculum and the public DEV evaluation, so the two are provably the same
  contract instead of two similar-looking ones.
* `h61.supervision` raises the loss weight of the coordinate tokens, which is
  the measured cause of the collapse onto a constant point.
* `h61.official_eval` measures the model against the upstream DragOn official
  public evaluation split, which is third-party and disjoint from TRAIN.
"""
