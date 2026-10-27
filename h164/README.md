# H164 — multi-original-application visual grounding recovery protocol

**Status: PREREGISTRATION + LEAK GUARD ONLY. No new model training or DEV/OOD evaluation completed yet.**

H163 is now a fully consumed external-app test: original MIT GridStack dashboard, **H157 0/16 → H161 0/16 genuine native browser task success**, despite valid H161 click JSON 16/16. The trained model was approximately **106.1px mean / 84.4px median** from the owner reference target center. This contrasts with H161 same-application new-scene DEV **0/24→24/24** and H162 familiar third-app native drag **0/16→16/16**. The failure is localization across a different UI implementation, not primarily command syntax.

## Frozen admission and licensing

- **Never** use H140 sealed blind, H158/H159/H160/H161/H162/H163 consumed evaluation screenshot SHA values or reference action gold during the new optimizer. `h164/protocol.py` checks the owner-private consumed SHA registry before admitting source data.
- Predeclare license, upstream Git full SHA, original UI implementation, browser proof, data source and case seed for each new TRAIN/DEV/OOD app. Copy licensed originals locally and preserve license/copyright.
- Full cohort minimum: **384 new TRAIN scenes** across **at least 4 original applications**; **96 new DEV scenes** across at least 2 applications; **64 independent OOD scenes** across at least **2 additional, previously TRAIN/DEV-unseen apps**.
- TRAIN must include native click, drag, and text input, plus both Arabic and English, and multiple randomized **TRAIN-only** sizes, padding, offsets, grid column counts, control placements, font scales and scroll depths from genuinely running apps. DEV/OOD must never upload reference actions or gold click coordinates to the model. Vary layouts enough that an agent cannot pass by memorizing one fixed toolbar position.
- OOD apps and source scenes must be frozen, reference-native successful and independently pixel SHA-identical **before candidate adapter inference**; lock one-shot consumed cohorts regardless of success.

## Hypothesis and free compute experimental arms

H161 performed a **language-LoRA-only** SFT with the visual backbone frozen. H164 should compare identical data/compute under two *predeclared* hypotheses, using free Kaggle T4 only if capacity allows:

1. **Control:** continue the exact H161 2B language-only rank8 SFT on entirely new diverse TRAIN scenes.
2. **Spatial arm (conditional)**: introduce a bounded trainable visual-projection LoRA or adapter, after a separate 1-step VRAM safety probe; no change to the flagship 35B. Do not claim that any proposed visual LoRA is feasible until actual T4 memory and optimizer state evidence exists.

Use fixed augmentation seeds, frozen base model SHA, same original app TRAIN cases for paired comparison, stratify by app/language/task/viewport and evaluate actual browser success rather than the fraction of JSON outputs that parse. Set explicit stop gates: nonfinite loss, leaked SHA, changed upstream source, paid GPU requirement, failed screenshot identity, non-original UI action, or VRAM cap.

## Independent decision gate

Report browser-native success per independent OOD app and task type, including click, drag and text, with matched model baseline/old/new paired outcomes, failures and abstentions. **No cross-app promotion** unless gain is reproduced on both additional OOD app implementations; with only 64 tasks, treat uncertainty and per-app failures explicitly. Never tune on consumed evaluation source, including H163. Only private owner evidence for images, model weights, reference positions and per-case raw model actions.

No publication, main merge, Hala 35B modification, H140 sealed BLIND calls, or paid services.
