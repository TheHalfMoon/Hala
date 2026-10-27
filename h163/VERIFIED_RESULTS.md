# H163 — VERIFIED negative transfer to a new original UI implementation

**2026-10-27, owner Windows + private FREE Kaggle T4. Final decision: no cross-app promotion.**

### External original app provenance

The third app from H162 was historically seen in H138. H163 therefore selected a **different original application not found in earlier Hala project source**: untouched MIT `gridstack/gridstack.js`, Git SHA `b05b79dcbca21caca046c1e8c4da6e59546ec806`, original upstream `demo/column.html`, and official npm `gridstack@14.0.0` package SHA-256 `c34298b66f84418a1c4e5130ad2fb6185ac1dea05c54e4223d64cefb2d524c2e`. Original repo and source/license stored privately; no upstream code is committed to Hala.

**16/16** new, bilingual original browser source tasks executed successfully with real mouse clicks and app state checks: 8 `set_column`, 8 `add_widget`; multiple initial grid column counts and two original demo layouts, eight Arabic and eight English instructions. Independently reproduced **16/16 exact full PNG SHA-256 images** in new browser contexts. Source manifest SHA `73f2416e65c3a843465d1dc2cce5614b91b050708b8e6d6388ba77add785423a`. All prior H158/H159/H161/H162 source screenshot SHA values excluded.

### Frozen GPU + true native reward

Model input-only original app archive SHA `e898cdfd25c814def7d52f92d4d15db1f2d51c685439449ad5ba8f36a4ad4150`; contains only the 16 images and their bilingual task instructions; **no gold reference click locations, DOM oracle or labels**. Frozen H157 research 2B adapter SHA `181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8`, trained H161 research 2B adapter SHA `72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c`.

Private free Kaggle T4: **32/32 true frozen model inference calls**, 16 per adapter, **zero optimizer steps**. Valid `click(x,y)` JSON: baseline H157 **0/16** → trained H161 **16/16**. **But this is only response formatting, not task success.**

Owner Windows original Chromium **32/32 independent, real UI native mouse interactions**, with pixel-exact source image match before each attempt. Actual original app native task completion: **H157 0/16 → H161 0/16**. The trained adapter failed all 8 `set_column` and all 8 `add_widget` tasks, both Arabic and English. Zero paired gains, zero regressions, exact paired sign p=1.0. Strict owner-private `h163/audit_model_reward.py` independently rechecked the frozen app/source/GPU/trial provenance and reconstructed the 0/16→0/16 result.

Owner-only posthoc geometry (not an accuracy or native-success score): trained H161's 16 predicted model click centers were **106.1px mean / 84.4px median** from owner-reference target centers; **0/16 within 25px** and **6/16 within 50px**. By task family, column change mean **145.8px**, Add Widget mean **66.5px**. Gold center distance is descriptive only; successful clickable regions are not necessarily 25px. This evaluation is **consumed**, so no tuning or rerun on the same holdout images, source states or owner gold positions.

### Decision and following H164 training requirements

**Do not promote H161 for general arbitrary computer use.** H161 truly improved same-two-app DEV 0/24→24/24 and transferred across two board implementations in H162 0/16→16/16, but the unrelated dashboard/controls layout remains unsolved. H164 must create fresh TRAIN-only source from **several additional original licensed UI application families** with diverse interactive control semantics and layout geometry, balanced tasks in both English and Arabic; reserve a distinct brand-new original application as an untouched OOD holdout, never feed H163/H162/H140 consumed images or reference clicks into optimization. Evaluate full native browser task reward once before any promotion.

Owner-private immutable evidence: `private-h163/source-once-v1/manifest.json`, `private-h163/H163_SOURCE_PIXEL_QA.json`, `private-h163/kaggle-output-v1/h163-free-t4-original-app-receipt.json`, `private-h163/H163_REAL_NEW_GRIDSTACK_BROWSER_MODEL_REWARD.json`, `private-h163/H163_POSTHOC_GEOMETRY_DIAGNOSTIC_ONCE.json`.

No paid services, H140 sealed BLIND calls, changes to flagship Hala 35B weights, media/weights publication, main merge, or release.
