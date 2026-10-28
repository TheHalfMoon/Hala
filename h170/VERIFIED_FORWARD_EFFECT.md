# H170: real frozen vision LoRA affects forward activations and logits

**2026-10-28, verified independent real free Kaggle T4 diagnostic.**

## Trigger and strict scope

H169 new original Chromium same-family DEV produced **H164 baseline 5/32 vs H164+H168 vision 5/32**, 0 paired wins and 0 regressions. All **32/32 raw generated action strings** (not just success counts) were byte-for-byte unchanged. This suggested either an adapter routing failure, or a smaller forward effect which did not alter greedy decoding.

H170 used **only one already-trained H168 original Choices.js TRAIN image** (case 168000) with the pinned original H168 32-TRAIN SHA. Absolutely no H169 DEV, H164 consumed OOD, heldout images, owner gold, or new native app task reward was used. Input prompt was stripped of `training_supervision`. Model weights were unchanged throughout. Private FREE Kaggle Tesla T4, pinned code commit `b61a807965963ade9305d376020087ed7d1a3fe6`, identical image and text in both forwards.

- Frozen H164 language adapter SHA `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- Frozen H168 visual-merger adapter SHA `506f571d0ff45bbe8932d9ea1f56dd6e4b5b5dc5f90cfdd1397243285523cd88`.
- Source original TRAIN archive SHA `d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6`.
- Private kernel `abdulazizshehri/hala-h170-private-vision-forward-sensitivity` COMPLETE, two paired forward calls, zero optimizer steps and no saved model.
- Independent `python -m h170.audit_forward --output D:\Hala-H170-Private\probe-output` returned **`H170_INDEPENDENT_FROZEN_TRAIN_FORWARD_DIAGNOSTIC_VERIFIED`**.

## Actual measured numeric effects

| Frozen model output under H164 language only vs H164+H168 vision | Measured delta |
| -- | --: |
| Visual merger `linear_fc2` max abs delta | **0.001953125** |
| Visual merger L2 delta | **0.0541410893** |
| Visual merger changed finite elements | **815,719 / 1,351,680** |
| Full-vocabulary final-token logits max abs delta | **0.0478515625** |
| Logits L2 delta | **3.848733902** |
| Changed finite logits elements | **234,394 / 248,320** |
| H168 trained vision adapter parameter norm sum | **2.343679368** |
| CUDA peak allocated / cap | **4.582 / 13.000 GiB** |

These are real nonzero numbers read from an actual T4 model forward hook and actual final-token logits, not synthetic toy tensors. Frozen language parameters did not change and the verified exact vision checkpoint was loaded. All numeric evidence is finite and cross-checked against two independently emitted private receipts.

**Conclusion:** the LoRA is **loaded, active and forward-effective** at both vision-merger and final logits. Its effects were **not enough to alter any of the 32 H169 greedy actions or native success rates**. This is NOT evidence that it generalized or improved real browser use. Nor does it uniquely prove why the action margin did not cross (limited samples, small update, rank, model prompting and frozen language could all contribute).

**Engineering decision:** keep H168 NO-GO for promotion, retain H164 frozen baseline. The next permitted training candidate must use substantially more *fresh, owner-controlled original TRAIN* examples spanning click/type/drag and bilingual instructions. A future blinded, newly captured DEV and unrelated new-app OOD are required after freezing model selection. Do not retest on consumed H169 or H164 benchmark data.

No paid compute, no public model/weights, no Hala 35B change, no main merge; Draft only.
