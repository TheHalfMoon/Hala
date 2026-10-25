# H125 — actual pretrained GUI-native 2B zero-shot comparison

**Status: FAIL for both frozen internal original H71 TRAIN-only drag gates. This is an independent, real neural research result, not a model-training result.**

## Private execution and genuine source controls

- New original 64 screenshot selection: 16 per real DragOn domain (sheet, slide_resize, slider, text_highlight); all original media byte SHAs and true START/END source target boxes independently qualified on owner free Kaggle CPU. Excluded by source ID and screenshot SHA all **2,560 previously used H105/H113/H123/H124 source images**, including all earlier heldout.
- Exact frozen source cohort SHA `81961961be80527d241850f6deb14ce9c9511833c85f58f2b3430b3d74eb6a48`; selected original media source digest `d95392f22d40b2a9c34a94421a790c94c5107e7bccff4b023df605b239c5ce49`.
- Genuine separate, private FREE noncommercial Kaggle T4 model inference, with zero H125 optimizer steps, no SFT and no public model/image publication. Both original model revisions immutable: UGround `ea7109f9573e5d8ee5170e425bd1fd73f2020e06` (Apache-2.0), ShowUI `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` (MIT). Each model generated two distinct START and END point outputs per source scene: 128 actual model point calls per arm.
- Original UGround tuple `(x,y)` on 0..1000 grid versus ShowUI list `[x,y]` on 0..1 relative grid were parsed as different native output spaces; original model-specific prompts and image/text ordering verified. Real source boxes NEVER supplied to the model as ground-truth input.
- Owner Windows **independent 24/24 verification PASS**, including true SHA-verified neural responses, source/case media/intent identity, frozen model revision, identical source image ID+image SHA for the two arms, and recomputed strict original START/END/FULL drag per-domain accuracy. This **only validates execution/evidence**, not GUI skill.

## H125 frozen original TRAIN64 strict drag results

| Preregistered metric | UGround‑V1‑2B | ShowUI‑2B | Required to pass |
| --- | ---: | ---: | ---: |
| Genuine source screenshots | 64 | 64 | 64 |
| Real neural endpoint calls | 128 | 128 | 128 |
| Native valid START and END outputs | 64/64 | 64/64 | >=48 |
| Strict START target-box hits | **0/64** | **1/64** | >=12 |
| Strict END target-box hits | **1/64** | **0/64** | >=12 |
| Strict BOTH points (full drag) | **0/64** | **0/64** | >=8 |
| Full drags in sheet | 0/16 | 0/16 | >=1 |
| Full drags in slide_resize | 0/16 | 0/16 | >=1 |
| Full drags in slider | 0/16 | 0/16 | >=1 |
| Full drags in text_highlight | 0/16 | 0/16 | >=1 |
| Unique native START points | 62 | 62 | >=32 |
| Unique native END points | 61 | 59 | >=32 |
| Modal START / END fraction | 3.125% / 3.125% | 3.125% / 3.125% | <=20% |

**Pre-registered internal gate: FAIL for BOTH arms.** Paired full-drag discordant cases zero and exact McNemar p=1.0: neither arm achieved any strict original full drag. The models both generated diverse valid native coordinate pairs but failed pixel-tight original source boxes. These measurements do not show which arm is better at *general* GUI skill, ScreenSpot or real computer workflows. Official model training/evaluation recipes and real GUI tasks differ from the original source's two-press-point drag contract. Do not misrepresent failure as general model inferiority.

## Implications and required scientific boundary

H123 (896-step END-only Qwen3-VL SFT), H124 (896-step joint START+END Qwen3-VL SFT), and H125 (zero-shot GUI-native UGround and ShowUI) all failed distinct stringent heldout gates. Repeating a small model with more steps or changing prompts after seeing withheld outcomes would be post-hoc selection. Instead:

1. Audit whether the task language and original screenshot make BOTH specific source and target bounding boxes *identifiable without oracle geometry* for each genuine drag family, and test this on a **NEW independent original TRAIN-only cohort**, not previously evaluated screenshots.
2. Separate output parsing, target visibility, semantic identifiability, coarse target quadrant, fine bbox localization, and actual drag execution into distinct preregistered metrics.
3. If the task is underdetermined from the screenshot plus task description, build a visual/interaction state representation or ask for the relevant UI state; do not train against a hidden target the model was never given sufficient information to infer.
4. Qualify a more suitable visual-grounding pretrained architecture / input resolution and genuine interactive trajectory evidence before any additional GPU training. Free compute only, no reuse of former heldouts for train/optimizer or model selection.
5. Keep old negative research outcomes immutable. No merge/release/deploy and no raw screenshots, annotations or model weights in public GitHub.

Owner-private source evidence: each private Kaggle notebook's exact attempt, summary, 64 real source cases, 64 neural response pair records; owner private independent report `h125-owner-independent-paired-neural-replay.json` (24/24 PASS). All private records remain on authorized owner systems.
