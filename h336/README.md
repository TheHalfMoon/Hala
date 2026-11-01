# H336 — Genuine pretrained Qwen first-click attribution (prospective, private)

**Status: code prepared and tested. No H336 model generation or Kaggle
job has occurred merely by creating these files.** A future distinct
owner-private free-only research run must be separately attested.

## Why this experiment, and why not another prompt change

The strict owner-private H333 T4 genuine pretrained Qwen source-exposed
study observed **EN 2/2 control and 2/2 candidate; AR RTL 0/2 for BOTH
control and candidate**. Every Arabic failure was the broader
`wrong_focus_or_state`. The H331 visual-target text prompt showed
no improvement. H326 had already demonstrated correct source-known
button-center conversion on 12 scenes, including RTL.

H335 built and genuinely qualified a first-party source-native browser
oracle to distinguish **five mutually exclusive first-click outcomes**:
`correct_open_and_focus`, `correct_open_wrong_focus`,
`wrong_station_open`, `no_station_open`, `abstain`.

H336 uses an independently pinned **pretrained** Qwen to generate one
first-click action per H319 public/exposed original scene. All 12
scenes are fixed, 6 EN and 6 AR RTL, alternated so one language is
not always first. Each case starts in a new offline 960×720 browser
context. Qwen sees only the screenshot plus task prompt, and the
trusted source oracle checks real native clicks and DOM/focus only
**after** the predicted action. No generated model point, source gold
or original image enters an aggregate output.

## Frozen prospective implementation

- Source repo GitHub `TheHalfMoon/Hala`, exact model-boundary commit
  `ff403d13249ac0f16d806ccfe4b81c4d92df6a07`. Pin is checked
  via actual Git detached checkout, without relying on moving branch HEAD.
- Original H319 HTML git blob
  `6e8467e123c7f0fbac77ea553622ab8e659f9c16`.
- Pretrained `Qwen/Qwen3-VL-2B-Instruct` exact revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Only the original `control` first-click prompt, original H323
  LM-format-enforcer model-time grammar and H273 normalized point
  conversion; **no prompt manipulation or candidate arm**. Exactly
  one wrapped model generation per source case, no second-step typing.
- Once a trained model can be independently legally admitted, the
  same category protocol can help design a *new* future evaluation;
  current H336 cannot serve as independent OOD and cannot prove LoRA.
- `h336/private_first_click_qwen.py` preflights actual owner-private
  noncommercial FREE Nvidia Tesla T4, available disk, dependencies,
  exact Git/HTML/model revisions and privacy before inference.
- `h336/build_private_payload.py` produces only one Python script
  and private Kaggle metadata, no images, weights, models, gold,
  datasets, receipts or other assets.
- The ONLY success output is a tightly bounded two-language table of
  five categorical counts, 6 per language, native click totals and
  12 claimed model first-turn generations. The output is never a
  model-weight identity attestation without independent owner-private
  Kaggle job verification.
- The ONLY failure output is a categorical stage (no raw exception,
  private media or model data).
- `h336/verify_private_first_click_receipt.py` refuses JSON
  duplicate keys, NaN/Infinity, extra fields, invalid source/model
  pins, impossible category totals, false TRAIN/OOD/adapter gain and
  unapproved file names. The returned safe summary includes only
  bilingual category totals, not individual predictions or points.
- Future unique private kernel target
  `abdulazizshehri/hala-h336-first-click-qwen-t4`. Do not overwrite,
  duplicate or retry without a new independently approved plan.
  A merely pushed/running kernel is not successful model evidence.

## Qualification and scientific limits

```sh
python -m unittest discover -s tests -p 'test_h336_*.py' -v
```

The CPU contract suite uses invented category values and mock policies:
these **cannot** prove any real Qwen behavior. H335 separately passed
real Chromium and Edge source browser qualification. H336 inherits
those source-native invariants but requires independent owner-private
model-time execution for empirical findings.

Before any prospective real GPU run: exact-head CI must qualify,
confirm owner quota remains free and eligible, no concurrent Kaggle
GPU job, no sensitive/private media, safe data rights for the
public/exposed diagnostic and matching frozen model protocol. If
blocked or unavailable, stop and mark `NO_GPU_SUBMITTED`. Any output
download must be restricted to the one allowlisted category report
or stage diagnostic, validate with the strict frozen verifier,
discard all extra files **without opening them**, and verify local
scratch deletion. Do not publish private receipts or output files.

**Release/learning gates still blocked:** H199 official TRAIN admitted
list remains EMPTY, H173 scored blind OOD SEALED, H212 barred,
H250/H251 third-party rights quarantined, H293/H294 private access
safety blocked; independent Google EGM Goldfish review not truly
completed. No trained-adapter gain or source-disjoint blind OOD exists.
No paid compute, weights publication, release/deploy or main-branch
merge is authorized by this research design.
