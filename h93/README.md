# H93 — REAL paired evaluation of original Qwen vs H92 on upstream DragOn official EVAL

Private personal noncommercial Kaggle research only; **no model training or publishing**. Source-first use of H61's existing pinned [official DragOn](https://huggingface.co/datasets/Hcompany/DragOn) **`eval/`** split, original upstream image/task loader, image framing, prompt contract, and box-geometry scorer. It does not access evaluation outputs during model prompts and does not change H61's Colab-only original runner/guards.

## Exact design

- H92 training is genuinely complete, and all eight original private optimizer checkpoint snapshots were independently downloaded and hash-validated on user Windows (18/18 checks). Final H92 adapter SHA256 `85fa3f6da1dd1b2ddd86d3db795ea19858b3e58bb12305abe2094c2a22e52497`, checkpoint SHA256 `6e3a2c678da49001e7309096ff33dcae8df70a0fa9a41c0cfd00b46ec0273d88`.
- H93 attaches *only* the private H92 notebook output as a readonly kernel source, loads the immutable H92 checkpoint with original H61 validator, checks the entire bound H92 training identity, and uses the real adapter in `PeftModel.from_pretrained(..., is_trainable=False)`.
- It samples at most **4 upstream official eval scenes per domain** across 4 domains (`slider`, `sheet`, `slide_resize`, `text_highlight`), which are disjoint from H71 `train/`. It runs pinned unchanged Qwen3-VL-2B first, then the same base with H92 LoRA on the **same images and prompts**. Deterministic generation, max 128 tokens/case, zero external action execution. Scores via unchanged `h61.official_eval.score`; reports base/adapter task-success counts and per-domain scores plus exact raw-output SHA and a private raw-attempt archive for audit.
- This is a **small public official EVAL diagnostic**, NOT an entire official benchmark, no hidden test, no comparison to other public SOTA, and **NOT bilingual Arabic/English evaluation**. English/Arabic quality remains unqualified until genuine heldout bilingual case execution.

## Kaggle and privacy

`h93/kernel-metadata.json` stays **is_private=true**, uses a free Kaggle T4, internet only for pinned public Qwen and upstream public EVAL source, and references only original private H92 output. The script `h93/kaggle_private_run.py` fetches exact reviewed source SHA `362b81374a6fcf74d45cd666434d74f94895f8a0`, removes optional incompatible `torchao`, pins Torch/Transformers/PEFT and records real provider-qualified status or fail-closed failure class.

Authenticated Kaggle CLI submission from user-owned Windows worktree:

```powershell
kaggle kernels push -p C:\Users\Shehr\work\Hala\kaggle-h93-private-run --accelerator NvidiaTeslaT4 --timeout 2400
kaggle kernels status abdulazizshehri/hala-h93-private-official-paired-qwen-eval
```

Results remain **private**, not uploaded to public GitHub. The existence of this source/CI alone is NOT proof that H93 inference executed; only real Kaggle output `hala-h93-private-eval/h93-private-eval-summary.json` with source SHA and private raw attempts can establish results. Do not claim any performance improvement if measured rates decline or are tied. No release, merges, or commercial use.
