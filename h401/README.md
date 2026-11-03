# H401 — first native-click paired evaluation of the private synthetic Qwen3-VL LoRA

This is a **source-only invented-image** evaluation of the H399 private 16-step LoRA experiment (Qwen3-VL-2B-Instruct, Apache-2.0). Unlike H399's teacher-forced completion-token loss, H401 attempts **actual free-form generation** of strict JSON point coordinates and measures real target-box hits, invalid JSON, and output position diversity. It evaluates **18 newly invented author-generated images**, 9 English instruction prompts and 9 Arabic. Images and targets are new relative to H399 TRAIN24/DEV6/alternate6. **This is not an independent sealed OOD test** and cannot establish real-world Hala GUI performance.

To prevent evaluating the wrong private weights, H401 pins the official model SHA `89644892e4d85e24eaac8bacfd4f463576704203`, requires exactly one mounted private owner H399 adapter, verifies its SHA-256 `8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537`, and separately checks H399's source training receipt SHA-256 `17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333`. Source notebook data comes from Kaggle's private H399 **kernel_sources** input, not copied public media.

The paired baseline and adapted runs use **the same 18 new image+instruction cases** and deterministic decoding, and record valid-JSON, exact target-box hit rates and number of unique prediction cells by EN/AR; invalid outputs score as misses. Only summary aggregates are saved. H401 is PRIVATE, Kaggle free Tesla T4, no externally linked datasets or public weights, Internet enabled only to fetch the official public Apache-2.0 model/pinned packages.

Kaggle kernel: https://www.kaggle.com/code/abdulazizshehri/hala-h401-invented-native-click-paired

## Verified Kaggle V1 actual native-point results

The owner-authenticated private Kaggle kernel **COMPLETED** and loaded the prior owner's private H399 adapter from Kaggle `kernel_sources`. The actual linked adapter SHA256 `8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537` and H399 original 16-step training receipt SHA256 `17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333` matched at runtime, before loading the base model.

The real image-conditioned **generated strict JSON native-click test** on 18 new developer-invented images returned:

| Metric | Base Qwen3-VL-2B | H399 16-step private LoRA |
|---|---:|---:|
| Overall correct click-box hits | **3/18 (16.7%)** | **18/18 (100%)** |
| English click-box hits | **0/9** | **9/9** |
| Arabic click-box hits | **3/9** | **9/9** |
| Syntactically valid JSON points | **4/18** | **18/18** |
| Distinct predicted cells (out of nine) | **2** | **8** |

This is evidence of learned **native-action format and colored-square grounding on a deliberately simple author-exposed, 18-case synthetic visual problem**. The after-adapter result does NOT establish generalization to real applications, different natural-language instructions, source-family-disjoint sealed OOD, safe abstention, open-world computer use, or the Hala flagship.

The original downloaded Kaggle `receipt.json` and exact `receipt.sha256` are preserved under `h401/verified_receipt/h401-private-native-paired-research/`, verified by stdlib `python -B h401/validate_receipt.py`. **Verified receipt SHA256:** `60a3dc447e4bede08b65a95edff21050feaa93a7df2ff82ee18660efda580328`. Four tests covering source originality and strict JSON/target geometry, plus four tests proving receipt authenticity and rejection of false OOD approval, passed on owner Windows. Nothing is asserted about unseen independent external evaluation or licensing for third-party screenshots.

**No actual Hala release clearance:** H199 approved GUI TRAIN digest registry still EMPTY; external legal rights review and signed H355 reviewer #384 and genuinely never-exposed external bilingual OOD custodian #385 still open. No protected main merge, no public weights.

 Hala flagship original GUI training and release remains **NO-GO** until independent rights approval (H199/H355, issue #384) and genuinely new external custodied source-family-disjoint bilingual OOD (H356, issue #385). Under no circumstances count this author-exposed invented evaluation as that independent benchmark.
