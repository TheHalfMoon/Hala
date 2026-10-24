# H107 — pre-registered independent strict 64-scene result

**No benchmark result is claimed here.** H106's GPU training must finish,
be downloaded privately and independently SHA-checked before running H107.
The previous H100/H101 evidence proves 0/16 strict successes for all
tested base/H99/prompt-only arms; source-only diversity is not ability.

## Design

- Original pinned independent DragOn public **EVAL** split (never TRAIN),
  four task types: slider, sheet, slide_resize, text_highlight.
- Freeze *previously unused* original EVAL sample indices **20..35** in
  each family = **64 individual source scenes**. H93/H95/H98/H100/H101
  already examined prior index ranges 0..19; do not reuse.
- Exactly three paired arms: Qwen3-VL-2B fresh base, verified original
  H99 vision-only rank8 128-step adapter, verified original H106 joint
  visual+language rank8 256-step adapter. Each starts from the same fresh
  pinned original pretrained Qwen base. Original H61 full-frame prompt.
- Genuine one-image inference with identical generation: do_sample=False,
  max_new_tokens=128, max_time=45, FP16 SDPA. Every domain 16 examples.
- Compare **native unmodified H61 strict scorer** on generated raw action:
  strict drag task success, start/end target hits, action JSON validity,
  degenerate start=end. A fenced-JSON repair is recorded only as an
  explicitly separate diagnostic; it must NEVER count as strict success.
- Exact source scene IDs+images+prompt SHA consistency across all three
  arms must hold. Raw outputs and source IDs stay ONLY in the owner's
  private Kaggle and Windows environment.
- Verify real H106 output against original checkpoint file-set SHA256,
  256 full optimizer steps, 256 nonzero finite gradients/real parameter
  updates, H71 source digest, H105 accepted TRAIN-plan+prefix SHA,
  exact joint visual/language module counts, signed config training SHA,
  and Kaggle's private H106 final receipt. Verify H99's independent
  pinned SHA and source as already qualified by H100.
- Owner independent replay after private download is mandatory.
  **No model gain, model readiness, release, or production claim** until
  strict GUI source-disjoint evidence is available. Even nonzero results
  on one small public EVAL batch must be reconfirmed separately before
  a broad model ranking.

Keep H105/H106 source immutable; Draft PR only, no paid services,
no datasets/weights published, no hidden TEST, no unchecked merges.

## Actual H107 owner-independent final result — 2026-10-24

Kaggle 3-arm 64-scene original DragOn public EVAL COMPLETE. Owner Windows
independently re-fetched all 64 scenes (pinned upstream commit,
4 domains × original EVAL indices 20..35), verified actual image SHA-256,
prompt SHA and raw neural output SHA, replayed original **unmodified** H61
strict scorer, and verified every arm/source pairing and Kaggle reported
per-domain scores. **All checks PASS, 0/64 strict successes for each arm**:
unchanged original base, original H99 visual-only and real 256-step H106
joint visual+language. Format-only repairs also yield 0/64 successes.

- Native strict action syntax: base **0/64**, H99 **0/64**,
  H106 **63/64**. This is *output serialization*, not GUI ability.
- H106 original strict start target-box hits **0/64**, end hits
  **1/64**, full correct drags **0/64**. 11/63 parsed drags had
  equal start/end positions.
- Private owner independently computed median start/end bbox miss of
  **115.55 / 170 pixels** for parsed H106 drags and median action distance
  **20 px** versus reference **117 px**. H99 diagnostic median predicted
  distance **0 px**; base diagnostic median **0 px**.
- Paired strict McNemar on 64 identical scenes gives **0 discordant
  successes** for H106 versus base and H106 versus H99. This cannot
  support a grounding improvement claim.
- All source IDs/raw output, original media, private evaluation receipts,
  real H106 checkpoint bytes and optimizer remain owner-only.
- **NO-GO for simply extending H106 256-step LoRA.** The bottleneck is
  spatial grounding, not native JSON. Next pre-registered experiment should
  target *explicit image-to-pixel positioning* (visual ruler/grid or
  controlled iterative focus) rather than prompt verbosity or more of the
  same completion-token loss. Any new candidate must use untouched later
  EVAL indices for a fresh, matched comparison.

H107 data isn't an independent hidden test. Do not claim global performance,
public release readiness, or a successful GUI agent from this result.
