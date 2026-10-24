# H116 — TRAIN-only start-conditioned signed END displacement

## Proven current frontier

H114 fresh Qwen3-VL-2B rank8 joint LoRA trained **256 genuine owner
verified optimizer updates**: 128 original H71 TRAIN source images,
128 START plus 128 independent END normalized-point supervision cases.

H115 evaluated 64 new original pinned DragOn public EVAL scenes and owner
Windows independently replayed every source-media SHA, all 192 neural
input/response SHAs, exact original H61 strict scorer and the actual
two-point coordinate conversions.

- H106 complete drag success: **0/64**, START 1/64, END 1/64.
- H114 complete drag success: **0/64**, START **15/64**, END 0/64.
- START matched 14 new-only wins and 0 old-only wins; exact two-sided
  McNemar p=0.00012207 on public EVAL. Real diagnostic START grounding
  improvement, NOT a successful computer-use drag model.
- H114 median START/END target-box misses: 34.59/112.75 source pixels;
  original H106 136.56/175.29.
- H114 median START-to-END model motion **6.11px**, vs reference
  **96.12px**. In 35/64 cases predicted START and END are within
  10px; in 38/64 predicted END closer to source START than source END.

All of these are **post-inference diagnostics**, not additional TRAIN
data. Never include the H115 source labels in H116 optimizer, prompt,
checkpoint, data selection or examples.

## Frozen new hypothesis

H114's one-point START localization has a real positive signal but
the independent END often collapses onto START. H116 introduces an
**explicitly conditioned destination task**:

1. On the ORIGINAL untouched H71 TRAIN full screenshot, retain the
   exact original H113 normalized START supervision unchanged.
2. For the END training example on the same unchanged screenshot,
   provide the authentic original TRAIN START normalized coordinate as
   a **proposed input anchor**. Output a strict signed `[dx,dy]` in
   native normalized image displacement units `[-1000,1000]`, where
   `original END = provided START + signed displacement`.
3. At future GUI inference, END's input anchor must be the **model
   predicted START**, never an EVAL/TEST label or oracle center.
   Any out-of-range coordinates, malformed/extra JSON, invalid
   direction or missing target mean no valid strict drag; never
   silently clip/repair.

This approach can still have teacher-forcing input distribution
shift between correct original TRAIN start anchors and imperfect
H114 model-predicted starts at inference. A future bounded original
TRAIN-only jitter-control ablation may help; no Jitter/EVAL tuning
or model gain is claimed here.

## What H116 actually qualified now

Using the owner-byte-qualified H113 private original derived JSONL
SHA `9a1bebc03c22825a6e0abef54edd2778f29e8144685884b811d95a7d8a7b02c9`,
the fixed original H105 512-source selection and H114 first 128
TRAIN-only image pairs, a NEW owner-private 256-row JSONL
contains **128 original START points unchanged +128 START-conditioned
END signed displacement targets**. No source screenshot or original
source-label bytes changed. Original 32 Arabic/96 English images and
all five original source families preserved.

Owner-private derived new JSONL SHA:
`174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c`.

The CPU script rederived and SHA verified the whole derived dataset,
checked all 128 true original pairs and nonzero signed displacement,
without any GPU, trained model, inference, public raw data, release
or paid service.

**Not yet executed:** a new fully separate model training run,
independent adapter/optimizer byte audit, real future source-disjoint
strict GUI comparison on untouched original EVAL indices 76 onward.
Do NOT announce improved END accuracy or working computer-use ability
from H116's source qualification alone. All branches Draft/private
and unmerged.
