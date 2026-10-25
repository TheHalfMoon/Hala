# H124 — completed negative original-TRAIN128 neural qualification

**Status: FAIL — do not promote or release H124 as a GUI-use model.**

## Actual research execution

- Owner private noncommercial FREE Kaggle T4 original H124 training **896/896** genuine optimizer steps, gradients positive for all 896; independently read back actual adapter/checkpoint/resume binaries on Windows **19/19 checks PASS**. Earlier 1,024 NEW H71 original TRAIN images and BOTH authentic START+END boxes real-byte SHA-qualified, exact 896 train and 128 heldout, no EVAL/TEST or previous 1,536 H105/H113/H123 source images reused.
- Model: Qwen/Qwen3-VL-2B-Instruct, pinned revision, fresh rank-8 vision+language LoRA; model predicts START, END, START cell, END cell, signed delta in one native JSON; neither endpoint supplied in prompts.
- Real private owner Kaggle inference evaluated **128/128 never-trained original H71 TRAIN heldout images**; 32 per four complex DragOn families. Owner Windows independently verified all raw neural response SHA, prompt/media SHA, exact model binary pins, original START and END boxes, and recomputed frozen evaluator: **13/13 PASS**. This pass is an **evidence verification** result, not a GUI-accuracy pass.
- Full provenance, individual source screenshots/bounding boxes and neural outputs remain PRIVATE. No source media, individual END/START labels or trained weights are committed or published.

## Frozen original-TRAIN-only heldout result

| Pre-registered criterion | Actual H124 | Requirement | Result |
| --- | ---: | ---: | --- |
| Native valid joint geometry outputs | **112/128** | >=112 | PASS |
| Strict original START target-box hits | **1/128** | >=32 | FAIL |
| Strict original END target-box hits | **2/128** | >=32 | FAIL |
| Strict original FULL drags (START AND END) | **1/128** | >=16 | FAIL |
| Full drags per family | **sheet 0, slide_resize 0, slider 0, text_highlight 1** | >=2 each | FAIL |
| Unique START points | **102** | >=80 | PASS |
| Unique END points | **99** | >=80 | PASS |
| Modal START / END fractions | **2.679% / 4.464%** | <=12.5% | PASS |

**Final frozen gate: FAIL.** Valid JSON and point diversity do not qualify model spatial grounding. Loss **1.7834 -> 0.5311** verifies optimization only and is not evidence of localization improvement.

## Private diagnostic only (aggregated; not model selection targets)

For 32 original heldout scenes per family:

| Family | True START 10x10 cell | True END 10x10 cell | START median normalized error | END median normalized error |
| --- | ---: | ---: | ---: | ---: |
| sheet | 4 | 8 | 86.10 | 73.77 |
| slide_resize | 5 | 0 | 220.20 | 341.79 |
| slider | 3 | 4 | 168.00 | 176.83 |
| text_highlight | 14 | 14 | 61.64 | 44.78 |

These aggregate errors are **post-evaluation diagnostics**. Never use the 128 withheld original labels or outputs to create H125/H126 optimizer data, adjust the frozen gate, cherry-pick benchmark slices or claim an unbiased subsequent evaluation. All 128 images must stay permanently withheld from future training.

## Conclusion / next research boundary

The START+END joint-coordinate hypothesis did NOT solve precise visual grounding at the source boxes. Repeating Qwen3-VL-2B with larger batches/longer SFT or post-hoc prompt shifts would not be grounded in improved evidence. Before another GPU run, compare more appropriate open-weight GUI-grounding backbones and image-resolution/patch tokenization, and qualify new genuine spatially diverse TRAIN media. Preregister a new source and evaluation before each experiment; independently measure real model-generated output against stringent original boxes. No deployment, merge, paid resources, model/image publication or release.

Evidence: owner Windows private receipt `h124-owner-independent-896-training-readback.json` (19/19), `h124-owner-independent-neural-replay.json` (13/13, final FAIL), genuine Kaggle private training and evaluation notebooks. The private individual data and binaries must remain on authorized owner systems only.
