# H326 — diagnose RTL first-click geometry without a GPU

**Original public-source oracle audit, not Qwen accuracy or model training.**

The genuine owner-private H324 pretrained Qwen3-VL-2B-Instruct
BeaconBoard trial completed: one English scene succeeded with a native
click then type; one Arabic RTL scene failed after the first native
click with categorical `wrong_focus_or_state`. Only one case per
language was observed. This does not establish generalized language
performance or isolate the root cause of the click failure.

H326 tests a **specific deterministic hypothesis**: the production
H273 0–999 relative coordinate conversion might misplace otherwise
correct screenshot-grounded button centers when the HTML page uses RTL.
Instead of spending GPU or changing the model's predictions, source
only obtains target button bounding boxes **inside the trusted
evaluator**, converts their centers into H273's prospective normalized
frame, and uses the unchanged actual production
`normalize_strict_action` to obtain pixels for the H263 native
browser actuator. Before dispatch, a real 960×720 PNG screenshot must
be identical across source snapshots. After dispatch, the independent
source oracle checks all four stations and the focused input.

The audit runs on the complete fixed set of H319 public-original
BeaconBoard scenes: 6 English and 6 Arabic RTL, 3 order variants,
both theme variants. All twelve known-center clicks must succeed.
It also tests six Arabic RTL negative controls in which **only** the
normalized x coordinate is intentionally mirrored. These must NOT
focus the correct input. A negative control failure stops qualification;
there is no posthoc coordinate correction or model intervention.

The source owns the reference coordinates and natural language target.
**Neither a Qwen model nor its policy receives these references.**
No screenshots, gold, points, task labels, model texts, private media
or model predictions are written to disk/GitHub receipts. Only
categorical public-source aggregate counts are returned. A successful
H326 will rule out a deterministic center-point conversion/RTL mirror
problem in these source cases, but it **cannot** tell whether Qwen's
Arabic visual localization, text comprehension, or target selection
caused the H324 failure. Follow-up model testing still requires
independent authorization, free compute and a preregistered protocol.

Local source-only unit command:

```sh
python -m unittest discover -s tests -p test_h326_rtl_source_geometry_audit.py -v
```

With authorized browser dependencies and environment
`HALA_H326_BROWSER_QUALIFY=1`, the real browser scenario also
runs. Windows Edge uses `HALA_H326_BROWSER_CHANNEL=msedge`.
GitHub Actions qualifies both original full-repository Linux Chromium
and Windows Edge. This audit does not admit TRAIN/DEV/OOD material,
does not run a model or optimizer, does not certify LoRA gain, and
does not release Hala.

H199 official admitted TRAIN manifest remains EMPTY. H325's
independent RailKit 32 examples remain TRAIN CANDIDATES requiring
external rights/source-history review. H173 historical scored OOD
sealed; H212 barred; H250/H251 third-party assets quarantined;
H293/H294 private retrieval safety blocks untouched.
