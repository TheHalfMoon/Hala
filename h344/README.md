# H344 — three-family source-only cross-visual comparison

**Research status: source-only, public/exposed, not TRAIN, DEV or blind OOD. No model-time inference, adaptation, or legal rights grant.**

## What this tests

The original offline source-browser paths produce exactly 32 RailKit, 256 SignalDesk, and 24 **public/exposed** BeaconBoard screenshot-stage frames. A trusted evaluator verifies the original first-party native click, focus, state and source image invariants before fingerprinting. It then computes 15,104 cross-family comparisons, with both decoded-RGB SHA-256 equality (exact visual duplication) and a **lossy 1,024-bit difference-hash distance histogram**. Raw images, digests, action labels, target coordinates, task prompts and per-scene outputs remain in RAM. Only aggregate pair counts and five fixed bins leave the source evaluator.

The fixed family-pair counts are RailKit × SignalDesk 8,192; RailKit × BeaconBoard 768; SignalDesk × BeaconBoard 6,144. Any decoded-RGB duplicate anywhere among the 312 images causes **refusal**, not a passing summary. A lossy difference-hash distance of zero can coexist with distinct decoded RGB; it must not be used to assert exact duplication, legal rights, semantic similarity, statistical independence or reliable model generalization.

The source families are not genuine source-family-blind OOD: all HTML files and their exposed tests are in the public repository. This study is exploratory contamination screening only. Source custody and independently authenticated rendered-image machine-learning training rights remain unapproved; H199 TRAIN remains empty. H173 sealed, H293/H294 restricted resources and quarantined third-party datasets remain excluded.

## Qualification

Run invented-only CPU contracts (no browser, model or private data):

```sh
python -m unittest discover -s tests -p test_h344_original_cross_family_visual_proximity.py -v
```

For complete source evidence set `HALA_H344_ORIGINAL_BROWSER=1` on an installed native Playwright Chromium or Edge. All **312 real source frames** and **15,104 inter-family comparisons** must pass without skipped scenarios, image substitution or generated data files. Workflow `.github/workflows/h344-original-three-family-visual.yml` runs the full source-only browser test on Ubuntu Chromium and Windows Edge. It uses no paid API or GPU, uploads no artifacts, writes no screenshots and cannot establish learned model improvement.

**Acceptance:** exact-head GitHub CI succeeds on both platforms with 312 source-derived frames, exact all-family RGB disjointness and bounded aggregate-only perceptual-distance reports; source roles still fail closed. Until the browser test passes, only invented-only test logic is qualified. Do not promote these numbers to observed visual similarity merely because the pair arithmetic is correct. Keep Draft and avoid merging or releasing.


## Safe observed aggregate output

The full source-browser test now prints one **H344_ACTUAL_PUBLIC_SOURCE_BANDS**
record only AFTER all 312 original source frames and all 15,104
cross-family comparisons have passed. The emitted record contains only
five nonnegative integer dHash distance-band counts per fixed pair of
public source families, the constant public cohort sizes, and explicit
FALSE TRAIN/OOD/model-gain/release flags. `source_only_public_band_receipt`
validates an exact bounded schema, impossible counts and any injected
raw image, point, gold or extra field fail closed. Results must be
read from the real exact-head Ubuntu/Windows source-browser CI run:
invented-only unit fixtures do not establish an observed histogram.

The 1,024-bit perceptual distance is **lossy** and can be zero between
distinct decoded-RGB images. Report only the actual numerical source
histogram and never turn a particular threshold into a claim of source
independence, legal screenshot ML use, blinded OOD or Qwen accuracy.
