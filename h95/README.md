# H95 — exact source-pinned fresh-sample strict DragOn GUI diagnostic

The original H93 provider run compared frozen Qwen base vs real 8-step H92 adapter on official public DragOn EVAL source SHA `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`: both **0/16 strict task successes**. Its generation samples were already inspected for model response formatting and geometry; they are not an untouched evaluation set for subsequent experiments. Secondary fence-only normalization yielded 15/16 base and 14/16 H92 valid actions but **0/16 successes** for either, with 7/16 degenerate drags each. This result must not be silently replaced.

H94 has now actually trained **64 additional** Qwen3-VL-2B LoRA steps on original owner-private H71 TRAIN, warm-starting H92 adapter with fresh AdamW, and owner Windows privately re-opened final real Kaggle checkpoint and validated **17/17 source/byte/gradient/optimizer checks**. Final H94 adapter SHA `d73cc9a302782bf7eca08f434a2050a9c3f212368c25afe65014f71319eb673c` and complete checkpoint SHA `c13fb4ce619318075006af3ac24f5fd0ce43757cb5746201c9d595328d3594ac`. Neither training loss nor a valid checkpoint by itself establishes better GUI task performance.

## H95 provider experiment

H95 compares **frozen H92 vs frozen H94** on a **different** preselected public official DragOn EVAL sample: images indexed **4–7** for each of the four original domains (16 paired scenes/model). H93 used indices 0–3. The upstream pinned source, exact original H61 image framing and prompt, real original task bboxes, raw Qwen decoding and **unmodified strict source geometry scorer** are reused. No TRAIN bytes are attached to the evaluation notebook. Both owner-private kernel checkpoint sources are admitted by SHA256, model identity, original TRAIN source and immutable optimizer state.

The official strict task-success score is computed directly on raw generated strings. A clearly separate secondary diagnostic removes **only exact Markdown `\`\`\`json` fences** and re-applies the same scorer; it is not labeled official task success. All generated raw attempts stay PRIVATE in owner-private Kaggle, with a SHA-bound aggregate report. This is a small public-EVAL diagnostic, **not** full official evaluation, independently blinded hidden TEST or English/Arabic bilingual qualification. No optimizers run in H95.

Provider notebook metadata `h95/kernel-metadata.json` pins both owner-private Kaggle kernels and real T4, no paid cloud or public release. Main launcher pins source SHA `21bddd6a30aaff536949257db60390e23f1533e7`.

```powershell
kaggle kernels push -p C:\Users\Shehr\work\Hala\kaggle-h95-private-run --accelerator NvidiaTeslaT4 --timeout 2400
kaggle kernels status abdulazizshehri/hala-h95-private-fresh-paired-qwen-h92-h94-eval
```

Accepted provider completion is NOT proof of model improvement until the actual result receipt and raw-hash-bound evidence are read back. Only compare the **paired same-new-scene strict results**, not H93 base results on a different sample. The scope is private personal noncommercial research only. Keep Draft/NO-GO and no model publication.
