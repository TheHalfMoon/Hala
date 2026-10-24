# H112 — Qwen native coordinate plane causal test (not new training)

## Independent research basis
- QwenLM/Qwen3-VL official repo documents relative grounding coordinates
  supporting boxes and points:
  https://github.com/QwenLM/Qwen3-VL
- QwenLM/Qwen3-VL official maintainers confirm normalized 0–1000,
  with source pixel recovery by point_x/1000 * source image width:
  https://github.com/QwenLM/Qwen3-VL/issues/1486
- Official ms-swift Qwen3-VL best-practice docs specify relative 1000
  grounding and train-time pixel→relative conversion:
  https://swift.readthedocs.io/en/v4.2/BestPractices/Qwen3-VL-Best-Practice.html

Original H61/71 and the actual H99/H106/107/108/110/111 drags used
**absolute source-image pixel coordinates** with `coordinate_space=source_image_pixels`.
The source prompt and LM's own pretrained relative coordinate semantics
may therefore differ substantially. This is a **falsifiable hypothesis**,
NOT proof that all Hala grounding failures stem from coordinate space.

H110 obtained 1/16 original strict drag with model-guided 512px two-crop
refinement, but H111 fresh frozen confirmation returned **0/32**
for baseline and refined arms despite 24/32 model-selected regions
covering both true targets. Independent owner readbacks agree.
Thus, longer SFT on original pixel-completion labels, a primitive grid
overlay or merely more untrained crop passes do not yet show reliable
grounding skill.

## H112 treatment, frozen before new EVAL
Four arms, original public DragOn **EVAL indices 56..59 per domain**
(4×4 = **16 source-disjoint new images**), unchanged actual pinned
Qwen3-VL-2B base and original byte-audited H106 joint adapter:

1. `base_pixels`: identical native H61 original source pixel prompt.
2. `base_normalized`: only prompt changes to model's pretrained
   relative [0..1000] x/y plane; same original screenshot bytes.
3. `h106_pixels`: identical H106 original H61 source pixel prompt.
4. `h106_normalized`: same H106 weights, native 0..1000 prompt.

Normalized output must be an **exact duplicate-key-free native JSON**
drag with fields `type,start_point,end_point,coordinate_space,ordered`
and `coordinate_space=normalized_0_1000`. Mapping independently
converts x using round(x*W/1000), y using round(y*H/1000),
clamping only 1000 to the right/bottom valid source pixel.
Malformed, fenced, unknown-space, float or out-of-bounds values fail
strict validation; NEVER repair or count those as success.

**Same original H61 score after deterministic conversion**: native
strict action valid, actual start/end target-box hit and full drag
success. The original H71 TRAIN media/labels, H106 checkpoint, LM
weights and original strict scorer remain unchanged.
No benchmark-answer coordinates in prompts, no hidden TEST,
no training or new model checkpoint. Only new original public EVAL
source images and prompt representation change.

The four arms use identical images and frozen model/inference settings;
the modified prompt may change format validity and model behavior,
and is part of the tested treatment. No new model is declared better
without independent original source/response SHA owner replay.
All source/raw neural answers private, noncommercial free Kaggle T4,
no paid services, publishing or merging. More than one source-disjoint
result would be needed for broad claims.
