# H105 — first actionable source-geometry diversity intervention for Hala

**Motivation:** H99 had 128 real optimizer updates but original source
selection was **81/128 (63.3%) one direction**, while all H100/H101 16-scene
strict geometry arms failed 0/16. H102 ruled out frame/oracle errors in
80 public EVAL cases and verified 128/128 actual H99 TRAIN media; H103
showed 2/8 original H71 image grids resized by the Qwen3-VL processor but
image tensors and coordinate completion masks matched 8/8; H104 optional
mapping remains independent, not used by this sampler.

Recent peer-reviewed GUI work prioritizes original training data quality,
spatial coverage, balanced sampling and a staged grounding-first curriculum
before RL. This phase is a **concrete new candidate data policy** rather
than a change to the H61 scorer or to an existing checkpoint.

- Deterministic *original H71 TRAIN 10k only* candidate selection, seeded SHA
  provenance and immutable source train labels; hard refuse EVAL/TEST.
- 512 distinct original screenshots (versus previous 128), chosen via
  direction (8 bins), displacement scale (3 bins), start/end 4x4 spatial
  regions, language quotas (target 30% Arabic) and five donor families.
- All source completion/start/end points verified inside exact original
  reference boxes and original pixel dimensions; one known half-open edge
  may be **quarantined for the proposed candidate only**, never changed in
  immutable source. Original H71 10k blanket admission remains unqualified.
- Compare *actual original H99* 128-prefix direction concentration against
  the proposed new 512-prefix source mix; candidate **only qualifies when
  dominant direction < 40%**, Arabic >= 20%, >=6 direction octants,
  >=5 source families, >=6 different start and end spatial cells.
- Source-private free Kaggle CPU: rederive H71 owner-private archive and
  full H61 curriculum, independently SHA-check and oracle-audit original
  media and labels of all proposed 512 selected examples.
- **No GPU/training/optimizer/weights/evaluation/hidden test.** This is a
  prerequisite to the *next* genuinely new SFT ablation; it does not prove
  model quality or claim success on GUI. Future experiment must use a
  frozen, new, disjoint verification set and unmodified H61 strict scorer.
- No raw images, original IDs, label points, selected ordered ID lists,
  model checkpoints or owner credentials in GitHub outputs. Only aggregate
  distributions, checks and a SHA of private deterministic plan.

Do not merge, publish, deploy or silently overwrite H99.
