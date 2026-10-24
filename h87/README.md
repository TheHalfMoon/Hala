# H87 — Real Qwen research training, free Kaggle + free Colab

User explicitly designated Hala's current GPU training work as **personal noncommercial research**. H87 introduces a dedicated `HALA_NONCOMMERCIAL_RESEARCH=1` runtime gate that only permits `platform=kaggle` inside an actual Linux Kaggle notebook. It does not change commercial release eligibility or the existing baseline's other runtime guards.

## Kaggle research target

`h87.kaggle_research_train.run` uses the **original private H71 DragOn 10,000-row TRAIN** archive, exact 1.76 GB SHA256 and the historical source Git SHA, via the unchanged H81/H67 full SHA and curriculum rederivation. All images retain original privacy and provenance checks. The script configures the original pinned Qwen3-VL-2B LoRA trainer for **one real optimizer step**, FP16, rank-8 adapter, gradient accumulation 1 and an explicitly bounded 32-record processor audit. The source and processor/model revisions stay immutable. The output is an independently reopened SHA-verified adapter-plus-optimizer checkpoint with one positive gradient/update receipt, if the provider can fit and execute it.

Private Kaggle runs must attach the existing H71 dataset `abdulazizshehri/hala-h71-drag-train-10000-private` and enable Hugging Face access to download pinned public weights. No datasets or training prompts are placed in notebook outputs. Checkpoint output is private; a true **independent private-provider readback** still needs separately verified after any successful training run.

## Status proof

The user-account Kaggle trial H86 completed on a real T4×2. H87 distinguishes genuine Hala LoRA work from that toy optimizer smoke. The research train receipt is not created on failure; failures are recorded separately. **Never infer trained Hala from an allocated GPU.** An authorized free Colab T4 is also available for a separate experiment and does not implicitly share the Kaggle private dataset or checkpoint state.

Keep Draft PR / NO-GO until actual first gradient-bearing checkpoint, independent output byte-readback, Arabic/English held-out GUI evaluation, and explicit publication decision.
