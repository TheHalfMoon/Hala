# H403 — private synthetic-only safe-abstention LoRA continuation

H402 established a real problem with the original privately adapted Qwen3-VL-2B (16-step H399) even on first-party toy colored rectangles: one unsafe false click and one malformed response out of twelve missing-target examples. H403 is a NEW follow-up 48-step rank-4 LoRA continuation specifically to improve missing-target abstention while retaining positive-box clicks.

## Data rights, separation and actual training

- **Base model**: the official Apache-2.0 `Qwen/Qwen3-VL-2B-Instruct` pinned commit `89644892e4d85e24eaac8bacfd4f463576704203`.
- **Parent adapter**: ONLY the user's private H399 16-step adapter pinned SHA-256 `8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537`, checked along with independent parent H399 training receipt `17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333`.
- **Original data**: 48 first-party algorithmically generated 224×224 colored-rectangle TRAIN cases, 24 DEV, 24 different-background author-visible AUDIT cases. TRAIN is never made from H402's prior evaluation cases. Every split is SHA-disjoint on complete pixels + UTF-8 instruction; all three are exactly balanced EN/AR and present/absent. New background, positions, bilingual phrasings and target coordinates; all made by `h403/kaggle/h403_safe_abstention_lora.py` in memory. **No third-party GUI screenshots, no private user files, no independent blind benchmark, and no patient data**.
- **Actual GPU**: PRIVATE owner [Kaggle H403](https://www.kaggle.com/code/abdulazizshehri/hala-h403-private-synthetic-safe-abstention-lora) on free Tesla T4, `48/48` genuine optimizer steps completed, 112 trainable LoRA tensor values changed relative to loaded H399 parent. The output adapter is 3,227,648 bytes, actual SHA256 `01f1c1846c76a5e1e785fcc431faed5ee2774638a0c46d84fedfbd2486ebcb4b`. Compared against the exact downloaded private Kaggle safetensors; the local inspection copy was deleted. The adapter itself remains **private in Kaggle only**, not in GitHub or a release.

## Paired generated-action result — limited author-exposed diagnostic

The diagnostic assesses **12 selected** original self-generated audit cases (six visible targets and six absent targets; each subgroup includes EN/AR), with the same H403 new audit inputs on the H399 parent adapter before H403 training and the new LoRA after training.

| Actually generated strict JSON action | H399 parent | H403 new adaptation |
|---|---:|---:|
| Total correct native clicks OR abstentions | **5/12** | **11/12** |
| Click the right invented target when present | **3/6** | **6/6** |
| Correctly abstain when no requested target exists | **2/6** | **5/6** |
| Unsafe false click when requested target absent | **1/6** | **1/6** |
| Invalid strict JSON | **6/12** | **0/12** |
| DEV loss (first 12 of distinct 24 DEV) | **0.464156** | **0.294818** |

**Important scientific and safety limit:** the unsafe false click is **not fixed**; one remains on the same six author-designed negative cases and arose in an Arabic condition. Improved JSON validity and synthetic click scores do not prove a safe agent. These are small developer-exposed synthetic images and bilingual template prompts, NOT real applications, independently sealed external OOD or full Hala.

The Kaggle owner original receipt and SHA-256 are preserved under `h403/verified_receipt/h403-private-no-target-lora/`, validated by source-only `python -B h403/validate_receipt.py`. Receipt SHA256 `ab607c8831aeec881ed7c22adc01e9fc8ff92a2a11fd09ec102aafe33d604165`. The verifier specifically rejects fabricated real-source, independent OOD or release claims and misreported EN/AR counts. Owner CPU source/test verification `11/11 PASS`, and isolated stdlib mode `7/7 OK` with one expected optional Pillow skip.

## Inviolable real Hala release gates

Official H199 legally approved GUI TRAIN SHA list remains **EMPTY**; H397 actual original H140 GPU entrypoints refuse unapproved source. Issue #384 independent original rendered screenshot and derivative model-rights review and #385 genuinely third-party independently custodied never-exposed source-family-disjoint bilingual OOD evaluation remain OPEN. H403 did not train the official Hala product model or prove action safety, no weights published, no protected `main` merge. **NO-GO for autonomous use and original GUI model release**.
