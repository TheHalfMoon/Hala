# H125 — GUI-native 2B grounding baseline, no fine-tuning first

**SOURCE-ONLY research until actual new-source CPU qualification; no H125 model
training, model inference, accuracy, published screenshot or release yet.**

## Why change

H124 real rank-8 fresh Qwen3-VL-2B joint START+END LoRA completed
896/896 optimizer updates, owner independently checked model artifacts
19/19 PASS. But 128 genuine original TRAIN-only heldout GUI scenes
scored **START 1/128, END 2/128, full drag 1/128**, and owner neural
replay 13/13 evidence checks PASSED while preregistered accuracy
criteria FAILED. All previous 128 withheld examples must remain out
of any future training/model selection. We should stop assuming another
generic 2B coordinate-SFT pass will solve the GUI grounding mismatch.

## Better native backbones to test FIRST (zero-shot)

- **osunlp/UGround-V1-2B** — Apache-2.0, Qwen2-VL 2B, trained for
  element location. Model card: https://huggingface.co/osunlp/UGround-V1-2B.
  Documented native output is **(x, y)** normalized 0..1000.
- **showlab/ShowUI-2B** — MIT, Qwen2-VL 2B GUI-action/grounding model.
  Model card: https://huggingface.co/showlab/ShowUI-2B.
  Documented UI grounding output is **[x, y]** normalized 0..1.
- **ByteDance-Seed/UI-TARS-1.5-7B** — Apache-2.0; a higher
  capacity GUI-specialized reference model, not suitable for blindly
  treating as an equally cheap free T4 baseline:
  https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B.

Licensing statements reflect Hugging Face cards; audit full dependent
software/dataset terms before any redistribution. A 2B parameter count
does NOT alone qualify T4 VRAM or guarantee model compatibility.
Model revision, processor, actual weights and max image tokenization
must be pinned before any real GPU inference; cap VRAM and abort if
not actually free.

## Independent source controls

1. Re-qualify H105 first512 + H123 full1024 + H124 full1024 original
   H71 TRAIN source schedule SHA. **Exclude ALL 2,560** images and
   labels, including prior heldout, by original ID and actual image SHA.
2. Select **64 new genuine TRAIN images**, 16 each sheet,
   slide_resize, slider and text_highlight, with support-aware direction,
   END region and scale balance. Verify raw screenshot byte SHA and
   BOTH authentic START+END boxes. Freeze case selection SHA.
3. Compare both pretrained GUI models on **exactly the same 64 images**.
   Each START and END is one independently prompted native point
   grounding query on unchanged original screenshot.
   Thus 2 model arms x 64 scenes x 2 endpoints = **256 real model
   inference calls**, if both arms pass runtime admission.
4. Parse native outputs **separately**. UGround pair (x,y) in 0..1000
   and ShowUI [x,y] in 0..1 MUST NEVER be silently interchanged.
   Use original source pixels to score each model-provided START/END
   against genuine START and END bounding boxes, complete strict drag,
   native validity and domain-wise counts.
5. Pre-register pass/fail for each arm and paired outcomes **before**
   running the model. No tuning based on prior H123/H124 withheld
   targets. Never equate ScreenSpot with stringent original DragOn
   6–11px boxes without an actual paired evaluation.
6. Only IF one pretrained GUI-native model materially outperforms
   the frozen H125 baseline, consider new **TRAIN-only** adaptation,
   never on the 64 withheld images, then a fresh independent holdout.

English-only complex-UI pilot. No paid GPU, no private image or weight
publication, no original EVAL/TEST, and no merge/release/deployment.
