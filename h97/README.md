H97 — 32-step private vision-LoRA study

H96 independently verified one REAL visual LoRA optimizer step over the original H71 TRAIN. Eight Qwen3-VL late visual attention modules and 16 updated vision LoRA matrices, gradient norm 1.320, measured peak VRAM 5.068 GB; 16/16 owner-host immutable checkpoint and optimizer SHA checks passed. NO downstream GUI quality improvement established.

H97 extends the exact visual-only configuration to 32 new optimizer steps on original private H71 TRAIN, with fresh optimizer and base pretrained Qwen. This does not warm-start from the H94 language adapter or H96's optimizer. Frozen language and base model, PEFT rank-8 on only eight last-four vision attention projection modules, coordinate-weighted supervision 8, no official EVAL data in TRAIN. One independent private Kaggle T4 job is allowed; no paid compute, release, publication or merge.

Pass requirements: 32 finite positive GPU gradients/losses and visual LoRA parameter changes, full immutable adapter-plus-optimizer checkpoint, real private provider receipt and independent Windows SHA readback. Training loss is NOT task success. Future evaluation must compare frozen H97 to Qwen base and H94 on new, disjoint official public DragOn EVAL scenes with original strict geometry scorer. Keep Draft/NO-GO.
