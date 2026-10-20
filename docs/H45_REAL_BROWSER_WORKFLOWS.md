# H45 — genuine browser interaction over original synthetic bilingual task pages

**2026-10-20. Scope: 24 locally authored fictional applications only, NO model inference/training, production OS clicks, real user accounts, external network or external benchmarks.** H45 builds on original H44 real screenshot [Draft PR #28](https://github.com/TheHalfMoon/Hala/pull/28) signed source `a14f88255b9a08549929c55735b787030e5e6614`. Do not alter H43 source, H44 24 original first-screen screenshots, H37 raw neural outputs, or H41 Kaggle CUDA running/queued experiment.

## What actually ran (Windows)

- Generated **24 original JS-enabled HTML pages** from unchanged original H43 bilingual 3-action state tasks: 12 English, 12 Arabic, 16 authoring-only tasks and 8 public DEV tasks, six fictional app labels and four visual layouts. Each fictional app renders selection, numeric quantity and a visible status line, with no network URLs or remote scripts.
- On **Microsoft Edge Chromium `Edg/155.0.4283.62`** headless via pinned cached `puppeteer-core 25.12.0`, the evaluation harness actually clicked the fake application's buttons and typed into its input. This is real *browser UI interaction*, not a Python-only action simulator, but only on deliberately synthetic pages—not Windows desktop/native user apps.
- Recorded for **each of 24 tasks**: untouched first frame; after selecting the intended item; after filling its quantity; after confirmation; one final frame after a *wrong item then quantity then confirmation*; one final frame after a correct plan followed by *unauthorized unrelated state mutation*. Total **144 actual browser PNGs**, 24 golden real-browser trajectories and 48 deliberately wrong real-browser trajectories. `tools/h45_edge_interactive.js` retains exact action arrays, per-action document state, visible text, roles, labels, bounds, PNG hashes, source HTML hashes, version and the refusal to contact external network targets.
- Genuine capture: `h45/assets/capture.json`, SHA256 `0d0b4d4f430d3d26eb7b4ce71318523e053bdd2293909a0fe73224a0b9ad2dd0`. Every frame is 960x540, CRC-valid PNG from browser. Create-only operation; no existing artifacts were overwritten.

## Separate Python oracle and no-answer leakage

`h45/audit.py` independently replays **every real-browser action** through original H43's source state machine. It compares every observed browser DOM state to its independently computed expected state, verifies the visible public status text, first-party source HTML/PNG exact hashes, visible/five-control nonoverlap, languages, and all action order. The independent model reference does **not simply match a golden action sequence**—success is based on the full desired state, and unrelated changes invalidate success.

Actual replay:
- **24/24 genuine browser golden interactions** reached allowed final task state.
- **24/24 wrong-target browser interactions** failed required postconditions.
- **24/24 unrelated-mutation browser interactions** failed despite reaching a superficially confirmed state.
- All **144/144 screenshot/DOM/state artifacts** recomputed and accepted as authentic recordings from this experiment; zero model outputs, actual production authorization or independent external app performance inferred.

Three create-only projections:
- `h45/assets/model_only.json` SHA256 `d8823c20ca8e2eb465faf879ecd06472c532a229adbd3c038d9d1b29b3d5a30f`: 96 image-only steps (four from each golden trajectory), containing PNG path/hash, current goal and public task metadata; **no target label, true actions, evaluator controls, expected state or success bit**. For any future image-only model comparison, feed only goal + PNG screenshot bytes, not the HTML, DOM API, page JS, original training manifest or evaluator-only JSON.
- `h45/assets/evaluator_only.json` SHA256 `6ce827761206fa8c2b763cf2fb9f7888dfd1f66ce64e74566f4e6316ce6e8cb6`: all 144 original-state checks, postconditions, controls, role/name/bounds and success, for an **independent evaluation process only**.
- `h45/assets/audit.json` SHA256 `4d2710f6c6de856ae4c101b76876d01c7fa41fbeaea8d15031ca0641f207c338`: provenance, all-case outcomes, and explicitly no model, native OS action or benchmark.

The embedded test HTML contains stable `data-ref` semantic element names, which are useful only to this *trusted first-party evaluator*. They are **NOT acceptable model input** and would make a model able to cheat. The browser DOM state itself is produced by the same first-party JS and is **not an externally authenticated real-app oracle**. The public DEV split is not sealed/blind TEST. These tasks still share one 3-step schema; no generalization to novel task dynamics shown.

## Reproducibility and release gates

```powershell
python -m unittest tests.test_h45_browser_interactions -v
python -m unittest discover -s tests -q
python -m compileall -q h45 tests/test_h45_browser_interactions.py
```

- H45 focused **4/4 CPU replay tests PASS**; all original file/evaluator hashes checked; tests refuse falsified screenshot hash, actual DOM state, final success, role/name/geometry overlap, browser privileged claims and missing frames.
- Pin source versions rather than silently modifying H43; browser pixels may vary with other browser/OS/font versions, so do not assert bit-identical universal rerendering. Browser screenshot originals are retained with SHA256.
- `graft build .` / `graft check .` wiring-only gate to follow successful serial tests; deep meaning tier not claimed.
- Source release remains **Draft/unmerged** until exact-head GitHub CI, substantive review and later independent training/quality/adjudication. No proof that Qwen3-VL-2B H37 baseline 32/64 improved and no competitor victory claimed.

## Highest-value H46

Implement **a consented true model baseline** over the H44/H45 pixel-only DEV projection with a preregistered prompt/checkpoint/version/generation budget and raw signed failures. Run **all** examples (never success-only) with a truly independent evaluator-only process and compare to the frozen H37 baseline under identical evaluation conditions. Improve grounding via licensed training/ablation only after memory/license qualification. Build truly distinct authored workflow families with modal dialog, scroll, async loading, error and real independent app holdout; verify none of the reserved tasks or official benchmark trajectories contaminate training. Actual H23 CUDA run, Arabic expert adjudication and production release remain separately unqualified.

## Measured Windows local engineering gates

- `python -m unittest tests.test_h45_browser_interactions -v` **4/4 PASS**, includes exact real browser frame checksum/role/state replay and deliberate image, state, wrong-target/unauthorized-operation and source-claim tamper cases.
- Complete serial `python -m unittest discover -s tests -q`: **558 tests run / 556 passed / 2 skipped**, exit 0 in **92.749 seconds**; original transcript `reviews/h45-full-suite-20261020.log` SHA256 `db1fd4cbeb2b11a908c7f73520f76ed570b97f55ec2477d6d6b56adeee341e1d`. `compileall` PASS.
- Source-specific genuine Graft **wiring-only** `graft build .` and `graft check .` PASS **1,736 nodes / 4,666 edges / 207 files**, graph in sync. **Deep meaning tier was not run**.
- Hosted exact-head GitHub Actions CI on Python 3.11/3.12, signed verification and any independent PStack/D1/Alibaba/Jev review remain separately pending; none should be represented as complete here.
- No paid compute, additional Kaggle submission, model neural weights, Mac or real desktop interaction. Original H44 files, `.ignore`, H41 CUDA notebook and all outside repositories remain unchanged.
