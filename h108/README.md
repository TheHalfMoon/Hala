# H108 — explicit on-screen original-pixel ruler: fresh controlled A/B

## Why change course
The owner-independent H107 exact original DragOn EVAL 64-scene comparison
has **0/64 strict GUI drag successes for all three** base, H99 and real H106
arms. H106 nevertheless produced **63/64 native strict JSON actions**,
whereas base/H99 produced 0/64. H106's median predicted displacement was
20 px versus ground truth median 117 px; median start/end target-box
misses were 115.55/170 px. The source/optimizer/model SHA, real H106
256 updates and benchmark replay all passed.

This is a clear **visual source-to-pixel grounding bottleneck**, not a
JSON-syntax or old dataset-integrity defect. Spending more free GPU
simply extending the same next-token objective is **NO-GO**.

### Research motivation — hypotheses, not claims
- Wang et al., *Improving GUI Grounding with Explicit Position-to-Coordinate
  Mapping*, COLM October 2026, proposes RULER position markers and improved
  spatial positional encoding for better patch-to-pixel mapping.
  https://www.servicenow.com/research/publication/suyuchen-wang-impr-colm2026.html
- Lei et al., *GUI-Spotlight: Adaptive Iterative Focus Refinement*, ICML 2026,
  proposes repeated visual focus refinement to improve grounding.
  https://proceedings.mlr.press/v306/lei26h.html

H108 is a **simple translucent image-space ruler baseline**, **NOT the
original RULER-token architecture, I-MRoPE, GUI-Spotlight or any claim to
implement those papers' full models**. Its purpose is to test a cheap
spatial-scaffolding causal hypothesis before investing in another SFT.

## Pre-registered new independent A/B

Original pinned DragOn **public EVAL indices 36..43** in each of
slider, sheet, slide_resize, text_highlight = **32 source scenes**,
untouched in all earlier H93..H107 inference comparisons.
**Four arms**, 32 paired scenes per arm:
- Original pinned Qwen3-VL-2B pretrained base, original image + prompt.
- Original pinned Qwen3-VL-2B pretrained base, image overlaid with a
  64-pixel semi-transparent source-pixel grid with 128-pixel labeled
  coordinate ticks, plus standardized short coordinate-grid instruction.
- Exact owner-byte-qualified 256-step H106 joint Qwen visual/language
  LoRA checkpoint, original image + original prompt.
- Same H106 checkpoint, grid-annotated image + same standardized
  coordinate-grid instruction.

For every source screenshot, the overlay is deterministic, doesn't read
or draw ground truth, preserves the 1024×640 dimensions exactly,
doesn't edit original image bytes, and **never transforms bbox labels**.
Each model receives the same source scene/prompts for its variant.
Actual image/prompt/response SHA recorded privately per case.

The old **unmodified native H61 strict scorer** measures exact drag
success, target start/end-box hits, native JSON validity, degenerate
start=end. A format-only repair is a diagnostic, **never GUI success**.

The result has **NO model training**, is **not a benchmark release** and
must not silently change the shipping Hala or the existing H71/H99/H106
checkpoints. Owner-private free Kaggle T4 only, zero paid resources and
no public model/raw outputs. If zero task successes, **NO-GO for ruler-only
approach**; prioritize source-bound staged single-point localization,
region crops, multi-resolution training and dense spatial supervision.
