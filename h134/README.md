# H134 original bilingual GUI START/END LoRA pilot

H133 showed an external frozen 2B grounding specialist hitting 5/8 START handles and completing 4/8 Chromium drags, versus ShowUI-2B 0/8. The paired sample was too small to prove general superiority. No Hala flagship model weights have changed so far.

## Original privately generated data

H134 fixture.html is a NEW owner-authored offline Chromium interactive task generator, not copied images or source content. For each scene it renders and executes an actual mouse drag, then records private genuine DOM START/END boxes, pixel centers, normalized 0–1000 coordinates and SHA-qualified PNG screens. Four GUI types: sliders, panel resize, table separators, card drag/drop. Every fifth task instruction is Arabic, others English, with randomized visual placements and distractors.

- TRAIN seeds 1000–1127: 128 screenshots, 256 START/END role examples. Private Kaggle dataset.
- DEV seeds 20000–20015: 16 source-disjoint screenshots. Separate private Kaggle dataset, only for candidate comparison before and after training.
- SEALED_TEST seeds 30000–30015: 16 different source-disjoint screenshots, OWNER WINDOWS ONLY, never mounted in Kaggle training.

All 160 source images were actually verified with Chromium and their labels checked against the real browser bounding boxes. No H71 medical/document source, original EVAL/TEST, H131/H133 screenshots, or external model datasets are used.

## Bounded free model optimization

Starting point: frozen pretrained external Apache-2.0 GUI Assist Grounder 2B, revision 31632edaa20749c7f532868a42635f7240033106. The experiment trains a separate tiny Hala research LoRA adapter, not the main Hala flagship 35B weights, and does not rebrand upstream weights.

- PRIVATE free Kaggle NVIDIA T4 only; no paid cloud fallback.
- Transformers 5.17.0, PEFT 0.21.2, Pillow 11.3.0, Kaggle torch 2.11.0.
- Vision/base frozen, language-only LoRA rank8 alpha16, and 24 REAL optimizer steps with gradient accumulation 2, batch1, fixed seed 134, AdamW lr1e-4, 640x480 images to bound T4 VRAM.
- Completion-only loss on pinned JSON point_2d 0–1000 grid (separate START and END prompts).
- Evaluate frozen model and new LoRA on exactly DEV16, measure START and END box hits independently before/after; sealed test NEVER attached to training.
- Save private adapter tensor SHA, model and dataset source revisions, loss and accuracy comparisons, actual optimizer steps and any resource problems.
- No model improvement can be claimed until actual GPU training and independent holdout verification; a pilot DEV increase alone cannot qualify production or Hala flagship generalization.

H127 independent blinded human target review of original H71 data remains unfinished, and that data is not involved. Keep all private assets and adapter owner-private and PR Draft; no release/merge/deployment.
