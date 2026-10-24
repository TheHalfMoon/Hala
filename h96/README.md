# H96 — source-bound visual-LoRA feasibility

H95 real 16-scene official-public EVAL: H92 0/16, H94 0/16 strict successes. H94 language-only LoRA improved JSON-format adherence, not GUI geometry. The old trainer expressly forbids visual parameters from receiving gradients.

H96 adds a strictly opt-in, one-step VISION-ONLY LoRA pilot on Qwen3-VL 2B: attention fused QKV and output projection in each of last four vision blocks (20-23), eight modules / sixteen LoRA trainable tensors, rank 8, FP16. All original pretrained Qwen weights frozen. H96 is separate from the trained H94 language adapter, with a different checkpoint identity and new optimizer; it cannot be used as a silent H94 warm-start or resumable combined adapter. Original language-only trainer defaults and checkpoint SHA identity remain unchanged.

The experimental runner reuses original private H71 TRAIN (hash verified and admitted), original H61 processor/masking/supervision, pinned model revision, original GPU gradient verification, immutable adapter+optimizer checkpoint writer and accurate local SHA readback. No Kaggle TEST/EVAL data are used for training.

Strict evidence needed: actual free private Kaggle CUDA allocation, exact 8 vision Linear modules/16 trainable LoRA tensors and no language trainable weights; nonzero finite gradients and changed visual LoRA parameters after first optimizer update; real persisted model+optimizer bytes; independent host SHA validation. GPU allocation or successful toy tensor code are NOT proof of training. Neither is training loss proof of better GUI accuracy.

Limits: one step, batch 1, rank 8, original selected images, no paid cloud, noncommercial personal research, private notebook and output, no model release, no merge until qualified. Further quality work may compare the frozen vision pilot with H94 under a separately reviewed combination/evaluation plan; do not assert performance gains now.
