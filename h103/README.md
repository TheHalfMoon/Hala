# H103 — actual Qwen3-VL image grid + coordinate label mask qualification

Source: original pinned private H71 10,000 TRAIN, rederived from the original
SHA-admitted immutable 1.76 GB archive. Use **owner-only free Kaggle CPU**,
no model weights, inference, optimizer, GPU, public data or hidden test.

## Why this stage is necessary

H100/H101 task success 0/16 for all variants; H102 proved the original EVAL
frame+oracle 80/80 and proved the actual H99 TRAIN samples/media 128/128, but
the entire TRAIN has one strict pixel bbox right==width edge (plausible
half-open convention) and is not blanket-approved. That does not demonstrate
that **Qwen3-VL's actual image processor** preserves the source coordinate
frame or that the trainer's completion-weighted mask hits the correct tokens.

## Fixed pre-registered probe

8 bounded, deterministic images selected from the *original* 10k TRAIN
using H75 bilingual/source-family selection, not sampled from EVAL.
A private processor-only load of the pinned Qwen3-VL-2B processor (exact
revision) then checks, for each image:

- Source screenshot integrity and its actual dimensions are preserved through
  the shared `training.scripts.train_adapter.prepare_inputs` path.
- Actual processor `image_grid_thw` * patch size equals the source width
  and height. No silent resize/crop/coordinate-frame drift is accepted.
- Full TRAIN and inference-style prompt-only preprocessing yield identical
  image patch grids and tensors; image/assistant prompt prefix IDs align.
- All instruction/image prefix tokens have label = -100 and loss weight zero;
  completion tokens match their labels exactly; coordinate digit tokens
  get the actual H99 weight 8.0, other completion tokens weight 1.0.
- Returns aggregate counts only; no screenshot, source ID, pixel tensor,
  task text, source completion, checkpoint, or hidden test in outputs.

**An actual PASS would only qualify the tested processor/label pipeline,
not Hala's GUI competence, general model quality or all 10k images.**
Any mismatch stops expansion/training and is investigated before a new LoRA.

No model architecture/training/release changes or publication.

## Actual owner-private H103 CPU findings (2026-10-24)

- Private Kaggle CPU run **completed its eight-case source-bound processor audit**
  and intentionally reported a **failed strict source-frame equivalence gate**.
  The failure is qualified evidence about the processor pixel grid, **not
  a model/weight training or inference error**.
- **6/8** exact source width/height preserved in the visual processor grid;
  **2/8** auto-resized to patch-aligned widths: **495×640→480×640** and
  **853×640→864×640** at actual **16-pixel patches**.
- **8/8** actual training-vs-inference prefix image grids/tensors identical.
  **8/8** original prompt/image-token prefix masking correct.
  **8/8** H99 numeric-coordinate token weight **8.0** and other supervised
  completion token weight **1.0** match their original labels.
- Exact immutable H71 TRAIN dataset SHA and Qwen3-VL processor revision pinned.
  Representative subset: 8 independent images, **2 Arabic / 6 English**
  across **5 source families**. Owner Windows independently replayed the
  provider's aggregate evidence **18/18** comparisons.
- The resize alone does **not prove a bug or explain Hala's 0/16 successes**.
  Qwen processing a patch-grid is expected; original-image coordinate-space
  can remain valid if conditioned on source dimensions. The H100/H101 EVAL
  frame is 1024×640, which was exact in H102, so don't claim a causal
  model error from these two source TRAIN cases.
- **Conservative result:** H103 image-grid identity gate FAIL as specified;
  masking and train/eval image-tensor identity PASS on 8/8. No data mutation,
  model training, paid compute, public outputs or PR merge.
- **Follow-up:** implement an optional explicitly invertible
  source-pixel ↔ processor-pixel mapping and test its strict-score geometry on
  the original public EVAL source; never silently reinterpret existing
  H71 labels or weaken the H61 strict scorer. Bigger risk remains the very
  small H99 training sample with collapsed drag directions.
