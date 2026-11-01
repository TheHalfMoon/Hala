# H335 — Diagnose the first native click instead of guessing at RTL

**Status: source-only diagnostic implementation, NOT an experiment on new Qwen weights or a completed independent model evaluation.**

## Evidence and hypothesis

The independent owner-private **genuine pretrained** H333 Qwen experiment tested 4 **publicly exposed** BeaconBoard scenes with 2 prompt arms (8 model-time runs total). Results after verified strict owner-only receipt: English 2/2 CONTROL and 2/2 H331 visual-target candidate, Arabic RTL 0/2 in BOTH arms, **zero paired improvement**. All four Arabic failures (2 per arm) were the broad `wrong_focus_or_state` category. They do not tell us whether a predicted click landed on the wrong station, fell on blank UI space or hit the correct station without focusing its input. H326 separately showed that the existing normalized→pixel mapping lands at the right *known* button centers in all 12 original English/RTL source cases, making simple RTL mirroring a weak hypothesis.

H335 adds a **trusted source-owned first-click diagnostic** that uses the same pinned H319 first-party BeaconBoard HTML, original isolated/offline Playwright browser context, real 960×720 screenshot consistency gate, H199 strict native JSON parsing, H263 native click, and H319 source-only oracle. It classifies a click into exactly these safe categories:

- `correct_open_and_focus`: intended original station alone opened and its input focused.
- `correct_open_wrong_focus`: intended station opened, but its input is not focused.
- `wrong_station_open`: different station opened (no multiple opens).
- `no_station_open`: click dispatched, but none opened.
- `abstain`: source state unchanged and **no native click** dispatched.

Impossible states (multiple open stations, nonempty text before typing, target focus when no station open, unrecognized station/focus, or any source/policy mutation) fail closed rather than being reclassified as success. None of the 5 categories contains coordinates or identities. Source DOM, expected target, focus, and states remain private **inside the trusted evaluator**; they are never inputs to the screenshot-conditioned policy.

`source_first_click` runs a single chosen scene and returns only a category with explicit flags that model identity, rights, blind OOD and adapter gain are **not attested**. `audit_first_click_source` runs **all 12** exposed original scenes through separate fresh source contexts and returns only bilingual category counts; no per-case model action/target, image, prompt, source gold, coordinates or model predictions are in the aggregate result. A callback may eventually be a genuinely preloaded pinned Qwen model, but this module neither loads models nor asserts GPU or model weights provenance. Callers that use only fixture-generated clicks MUST NOT report model performance.

## Evidence and tests

`tests/test_h335_original_first_click_attribution.py` checks five bounded categories, adversarial impossible source states, role denial, false claims and 6-en/6-ar aggregate counts. With `HALA_H335_BROWSER_QUALIFY=1`, it uses real **Ubuntu Chromium / Windows Edge** with all 12 source-owned original button center first-clicks plus original two-language wrong-button, blank-space miss and abstention negative controls. Those clicks are **author/test oracles, NOT model predictions**; the positive 12/12 is browser qualification, not Qwen performance. HTML/PNG/source state remains in memory.

Owner Windows source-only CPU:

```sh
python -m unittest discover -s tests -p test_h335_original_first_click_attribution.py -v
```

This task cannot infer the exact cause of past H333 failures without re-running the model with the new instrument in a separately reviewed privacy-safe protocol; H333's raw private model actions are not recovered. Since the H333 prompt-only variant had no benefit in its tiny exposed cohort, avoid more identical prompt-only GPU re-runs. A broader scientific study first needs lawfully cleared TRAIN material (H199 official admitted TRAIN set remains empty), new source-independent DEV and fresh never-exposed blind OOD, and a true paired BASE-vs-ADAPTED study. An H335 diagnostic on this same public source would remain exposed exploratory research.

Google Elephant–Goldfish-style H330 design is a **proposal**; no authentic fresh-context Goldfish reviewer has completed a review yet. Keep H173 scored OOD sealed, H212 barred, H250/H251 assets quarantined, H293/H294 owner-private access blocks unchanged. Zero cost; no new Kaggle run, GPU, model training, source admission, raw media publication, weights, deployment, release or main-branch merge.
