# H99 — original H71 TRAIN source, balanced visual-grounding pilot

The original H71 curriculum contains **10,000** SHA-shuffled training records, and the real H94/H97 optimizer receipts used only 64/32 of them (0.64% / 0.32%). Their separate 32-record processor audit is not the same as the actual optimizer-exposed training records. H98 authentic three-arm official public DragOn EVAL measured **0/16 task success in original Qwen, H94 language LoRA, and H97 vision LoRA**. No performance improvement is established by lowering TRAIN loss.

H99 is a copy-first, **opt-in** sampler layered onto unchanged `train_adapter.train`; previous default training, model scope, resume identity, manifests, H71 record bytes, score definitions and frozen pretrained source remain unchanged.

## Sampling invariants

* Only original source-bound `split=train`, valid source pixels drag completion, SHA-verified original H71 dataset and the H61 family/language metadata. No official EVAL/TEST enters training.
* Deterministically group all eligible source rows by **(language, family)**, then interleave across (drag octant, drag distance bucket) with a fixed seed and SHA-stable within-bucket ordering. Select visually unique source screenshot hashes; exclude source **start=end** no-op drags.
* Refuse unqualified records, duplicate IDs, invalid geometries, unsupported sampling mode, unavailable language, too few unique images or inability to cover each language/family stratum.
* **Exact chosen training IDs** are recorded only in *private original optimizer checkpoint microbatches*. Public aggregate receipt publishes only group/language/direction coverage counts and the SHA-256 of the complete ordered private ID plan. Re-derivable preflight and actual per-step GPU optimizer sample IDs MUST match the plan.
* SHA-bind sampler schema+plan digest to a separate training identity, so old language-only and old vision-only checkpoints cannot silently masquerade as balanced training. The old default train call (`sampling_strategy=None`) stays byte-for-byte equivalent in selection and signature.
* Guard exactly **128 new, positive-gradient visual LoRA optimizer steps** with fresh AdamW, pretrained pinned Qwen3-VL-2B and original H71 TRAIN source. Exactly 8 late visual attention modules, 16 trainable LoRA tensors; base and language frozen. Strictly zero paid GPU and personal noncommercial, private Kaggle T4 only.

This is an *exploratory source-diversity experiment*, not proof that either geometry accuracy or bilingual behavior improves. Assess afterwards on another disjoint pre-registered original DragOn public EVAL sample (beyond previously evaluated indices 0–11/domain), with unchanged strict scorer and separate fence-only diagnostics. No release, public model or PR merge.

CPU tests: `python -m unittest tests.test_h99_balanced_train_sampler tests.test_h99_source_bound_visual_study -q`. The private owner Kaggle runner is source-pinned once H99 core passes tests and is committed.
