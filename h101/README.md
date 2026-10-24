# H101 — frozen H100 geometry-error follow-up, no model training

H100 original DragOn strict replay passed 65/65 checks with **0/16** strict successes in every arm. Independent private H101 residual analysis showed H99 median target-box start miss ~100.7 pixels, end miss ~167.2 pixels, median predicted drag displacement 13 px and **6/16** degenerate starts=ends. Baseline and H97 predictions likewise had no strict hits. Training loss or formatted JSON cannot count as geometric success.

**Pre-registered isolated causal probe**: same base Qwen3-VL-2B and exactly source/optimizer-admitted H99 visual128 on 16 NEW upstream official DragOn public EVAL scenes, fixed indices **16..19** across the four original domains. Four matched frozen arms: base+original H61 prompt, base+geometry instructions, H99+original H61 prompt, H99+geometry instructions. Every arm freshly loads the same pinned base; no LoRA training, stacking or calibration. Original H61 prompt is unchanged for controls; intervention appends solely dimensions + generic visual localization guidance, NEVER target boxes or solutions. Strict H61 geometry scorer unmodified. Exact fence repair only separate format diagnostic. Full owner-only raw case outputs and source images stay Kaggle/Windows private; aggregate results can enter GitHub.

This is public EVAL **development evidence**, not hidden TEST, deployment/bilingual qualification or valid claim of superiority. A post-intervention strict improvement, if any, is only exploratory; use another untouched source set for confirmation. No paid GPU, no public model, no merge or release.

## Actual owner-private H101 outcome (2026-10-24)

Kaggle owner-private run `abdulazizshehri/hala-h101-private-fresh-coordinate-prompt-ab` **completed** on two Tesla T4 GPUs. The owner downloaded raw neural outputs and the provider summary and independently re-fetched the original upstream public EVAL scenes. Owner-only `kaggle-h101-audit.py` independently replayed image/prompt/output SHA-256 and the exact original H61 strict scorer on the fixed 16 new cases: **80/80 checks PASS**. No model weights, prompts with raw answers, images, or raw private evaluation outputs were pushed to GitHub.

| Frozen arm | Strict GUI successes | Strict native JSON validity | Format-only validity | Format-only degenerate drags |
|---|---:|---:|---:|---:|
| base original H61 prompt | **0/16** | 0/16 | 10/16 | 5/16 |
| base generic coordinate prompt | **0/16** | 6/16 | 12/16 | 6/16 |
| H99 original H61 prompt | **0/16** | 0/16 | 11/16 | 5/16 |
| H99 generic coordinate prompt | **0/16** | 11/16 | 14/16 | 8/16 |

**No start-box or end-box hits in any arm, even after format-only normalization.** Strict task-success paired effect is exactly **0** for both base and H99 on this small set. Generic localization prompt improved native JSON, **not visual grounding**; in H99 it also increased degenerate drags. No qualified computer-use improvement.

**Research decision:** H101 prompt-only geometry intervention is a **utility NO-GO**. Next source-bound work should inspect actual visual token/image resolution and supervision alignment and design a TRAIN-only coordinate-grounded, nondegenerate candidate with a frozen genuinely unseen comparison. All gains must be measured on unchanged strict geometry, not loss/JSON counts. No continued H101 reruns, public release, merges or paid compute.
