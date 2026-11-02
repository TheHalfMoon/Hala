# H352 — Screenshot-only button-center calibration, paired genuine-Qwen pilot

**Engineering/research Draft. NO H352 GPU submitted or result yet.
Not an adapted model, not blind OOD, not an authorized TRAIN manifest.**

## Why this intervention exists

The completed owner-private H351 real pretrained Qwen3-VL-2B first-click
pilot on the original first-party PUBLIC/EXPOSED H319 BeaconBoard gave
EN 6/6 correct first-button/focus, AR RTL 4/6 correct and **2/6 with
12–48-pixel miss distance** from the requested button. This is
stronger evidence than inventing a coordinate explanation: the source
evaluator measured misses using native post-click state and trusted DOM
rectangles, but only broad counts were exported. It does not reveal
their direction or per-case points and does not imply a calibrated fix.

H352 preregisters one falsifiable intervention: **after** the exact
same pinned Qwen model generates ONE native first-click action from
the original screenshot, a fixed computer-vision rule detects four
large solid-colored button-like rectangles using **screenshot RGB pixels
alone**, and snaps a near and unambiguous point to the nearest detected
button's center. It refuses unsuitable 960x720 images and never
chooses a target using DOM/scene/gold/task text. Outside 48 pixels or
within an ambiguous neighborhood, it leaves the original action.
Abstention is preserved. The threshold is chosen as a hypothesis
from the already observed H351 12–48px band, not tuned to hidden OOD.

This is intentionally **H319-specific public-exposed sandbox research**,
not an approved policy for live healthcare systems or arbitrary
websites. No unbounded probing, additional risky clicks, external
URLs, real records or clinical decisions.

## Real source-only evidence now observed

On owner Windows Edge, H352's purely image-derived rectangle detector
found **exactly four** button regions in all **12 original H319 scenes**
(EN/AR, light/dark), and each detected center fell inside exactly one
original clickable station button when separately checked by the
trusted **test-only** DOM evaluator. Source DOM is not passed to the
pixel policy. Eight invented-only CPU tests passed as well; no Qwen
was run for this local detector qualification.

## Prospective true paired model-time protocol

`h352/private_qwen_paired_snap.py` is a code-only Kaggle T4 study
which uses a single frozen pretrained Qwen first-click generation and
model-time grammar per scene, then opens **two independent identical
native browser source scenes**: baseline executes the original Qwen
native click and candidate executes the pixel-only snapped action.
For source scoring only, trusted original DOM/state/focus is consulted
AFTER dispatch in each arm. Both arms share exactly the same
model-generated action and exact decoded RGB image identity, so Qwen
generation random drift cannot masquerade as calibration gain.

It reports only low-entropy bilingual aggregate baseline successes,
candidate successes, snap-change counts and paired win/loss/tie totals,
with strictly validated arithmetic and no per-case images, points,
model output, task identifiers, source oracle, weights or logits.

Source: original `h319/beaconboard.html` Git blob
`6e8467e123c7f0fbac77ea553622ab8e659f9c16` at exact H351
parent Git commit `06596fdd70214be8f6729eb94615f569d9a7ec85`.
Model `Qwen/Qwen3-VL-2B-Instruct` pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203`.
`h352/build_private_payload.py` combines the public pinned snap source
and paired private entrypoint into ONE uploadable private script with
private Kaggle metadata, no model/media/datasets. It never submits
jobs itself. Distinct fresh kernel:
`abdulazizshehri/hala-h352-paired-pixel-snap-qwen-t4`.

## Qualification (never model-score evidence)

CPU unit tests:

```sh
python -m unittest tests.test_h352_pixel_button_snap tests.test_h352_private_paired_snap -v
```

Actual source screenshot/DOM QA (not Qwen inference):

```sh
HALA_H352_ORIGINAL_BROWSER=1 python -m unittest tests.test_h352_pixel_button_snap -v
```

Dedicated GitHub Actions tests CPU on Ubuntu/Windows Python 3.11/3.12
and true native source browser on both platforms. Exact-head hosted CI
must pass before any private inference submission.

## Strict NO-GO conditions

- Source is PUBLIC/EXPOSED and synthetic. Even a paired correction
  improvement would be **system-level geometric postprocessing on a
  known application**, not learned model generalization or blind OOD.
- No H199 official TRAIN admission; its registry stays EMPTY.
- H343 independent rendered-image machine-learning TRAIN rights review
  has no independent reviewer decision; no permission is inferred from
  Apache-2.0 HTML metadata.
- H173 OOD remains sealed, H212 barred, H250/H251 quarantined,
  H293/H294 blocked.
- No Qwen weights, individual clicks/coordinates, screenshots, private
  output, training examples or personal data in this repo or PR.
- A future one-time private, free-only GPU run requires exact-head CI,
  fresh actual Kaggle free quota, no existing duplicate kernel/session,
  privacy/rights preflight, code hash provenance, and strict private
  category receipt verification. **Do not claim H352 improvement before
  a genuine completed and authenticated paired model run**.
- No merge, public release, paid service, or production clinical use.
