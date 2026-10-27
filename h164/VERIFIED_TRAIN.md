# H164 — 384-source TRAIN admission VERIFIED (2026-10-28)

**Result: TRAIN data source/browser qualification PASS. Neural weights are NOT trained yet.** Separate original DEV/OOD source collections are required before any optimizer steps.

## Exact owner evidence

- First 96 TRAIN source: 24 each original unmodified TodoMVC, React dnd-kit Kanban, SimpleTable and Calculator. **96/96 independent full PNG byte-exact Chromium replay**; manifest SHA-256 `887eddd886d7f0505fcca4afd79ee96ce0dadf2443c72c0e5932f418e751d577`. Receipt: `private-h164/H164_TRAIN_PILOT_PIXEL_QA.json`.
- Additional 288 TRAIN source: 96 each original SimpleTable, Calculator and SortableJS. **288/288 independent full PNG byte-exact Chromium replay**. Final frozen manifest SHA-256 `d2d2b8231a2995fb2b9e7c0d9ee4048c5e21152fa558103e344dc10697572104`. Receipt: `private-h164/H164_EXPANDED_288_PIXEL_QA.json`.
- Complete 384 combined **unique screenshots**, across **5 actual original application implementations**, and **192 English/192 Arabic**. New 288 actions: 160 click, 96 drag and 32 type; including first 96: **224 click, 120 drag and 40 type**. Owner independent contract/source audit `h164/audit_expanded_train.py` passed; 144 previous consumed evaluation images and 96 first-pilot source images excluded.
- Original upstream Git commits and license statements are pinned by the source collector. No donor code was added to Hala's GitHub repo; all PNGs and their reference coordinates are private on owner Windows.

## Why two earlier copies were rejected, and what was done

The first partial table pilot (56 images) had duplicate images and was quarantined. The first 288-candidate TRAIN data then failed independent re-screenshot at Calculator case 132. A recapture with parked mouse still failed case 105. **Neither failed cohort was ever used in training or benchmark selection.**

The original Calculator demo draws its equals button with a blurred GPU shadow that produced nondeterministic 8-pixel edge changes in otherwise identical fresh Windows Chromium screenshots. For **H164 new TRAIN and DEV capture only**, the collector now parks the mouse, disables button transitions and normalizes that equals-button shadow using a transient browser-injected stylesheet. This changes the rendered pixels but **does not change the donor's source code or the actual native click behavior**. The source manifest explicitly records `calculator_shadow_normalized=true` and rejected source SHA. Both the current source verifier and independent owner manifest auditor require these flags. The corrected final 288-case source then passed **288/288 byte-exact replay**.

## Downstream qualification

The H164 protocol still requires **96 fresh DEV images** with no owner gold labels in model input and **64 fresh OOD scenes** from two truly different original applications never seen in TRAIN/DEV. Pell rich-text editor and Pikaday date-picker are the selected independent original MIT app candidates. All source scenes must be created only once, with reference-native success and fresh browser pixel-exact qualification, before packing model inputs.

The H164 Qwen3.5-2B rank8 external research SFT module is SHA-locked to `UNQUALIFIED` until both DEV and OOD pass. Its predeclared run uses **192 optimizer steps, batch accumulation 2, one exposure of each of 384 TRAIN original screenshots** on **FREE Kaggle T4 only**. No H140 sealed BLIND, no Hala 35B weight changes, no paid GPU, no media or model weight public release, no main merge. Keep Draft.
