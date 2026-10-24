# H117 — START-conditioned signed END delta LoRA

## Measured reason for changing the objective
H115 evaluated 64 new original DragOn EVAL scenes. H106 scored 0/64 strict drags; H114 also scored 0/64, but improved START hits to 15/64 while END remained 0/64. An independent Windows replay reproduced the source-media, prompt, model-response and original strict scorer outcomes.

The H116 posthoc diagnostic found a median real TRAIN start-to-end separation of 153.5 px, versus 6.11 px in H114 predictions. On old EVAL, the model's predicted END was closer to actual START than END on 39/64 cases. **These EVAL cases are diagnostic only and excluded from training.**

## Frozen training hypothesis
Use the original qualified H113 512 TRAIN-image/1024-point source, selecting the first 128 original paired images under H105. Create 128 identical START point tasks and 128 END signed movement tasks, where the authentic original TRAIN START normalized coordinate is input context and END target is a signed relative displacement. The screenshot remains original and full-frame. The parent H116 derivation is SHA-qualified.

Run exactly 256 fresh rank8 FP16 LoRA optimizer updates on Qwen3-VL-2B. Eight late visual layers and original language q/v LoRA layers remain the same as H114. No H114 or H106 checkpoint is used to initialize this isolated pilot; previous training pathways are preserved.

**Distinct improvement to the training implementation:** prior coordinate-weighted loss overemphasized numeric metadata in coordinate_space strings (the zero and thousand digits), not just actual position tokens. Only in H117, coordinate weight 8 applies to genuine START point values or signed END delta values, including the sign; metadata receives ordinary weight 1. CPU tests enforce this distinction.

## Known risk and gates
The authentic START anchor is present in TRAIN prompts, but future inference must use the model's own predicted START. That introduces a source distribution mismatch and may reduce performance. Do not introduce evaluation labels as prompt anchors or use EVAL to tune this model.

Pin the original private H113 derived SHA 9a1bebc03c22825a6e0abef54edd2778f29e8144685884b811d95a7d8a7b02c9 and H116 signed-delta SHA 174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c. Verify all 256 role IDs, finite nonzero gradients and actual parameter updates, full adapter/checkpoint/optimizer SHA and independent owner readback. Lower loss or valid JSON does NOT prove GUI improvement.

Next original strict GUI comparison on untouched public DragOn EVAL indices 76–91 in each of four domains (64 screenshots) should compare H114 independent absolute points, H117 START plus conditioned signed delta, and frozen original H106. All models must use unchanged original H61 scoring, and identical media, with proper model-call-cost accounting.

Private, Draft, unmerged; no publication, paid GPU/service, model-weight release or deployment.
