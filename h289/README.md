# H289 — RouteBoard: disjoint original DEV-only research GUI

**A new independently authored, public, offline bilingual development
research family, NOT training data, not a sealed independent blind OOD,
not officially admitted DEV and not a trained Qwen result.**

## Purpose

ShiftDesk H285/H286 contributes 24 original author-owned **TRAIN
candidates** (12 English / 12 Arabic) aligned to Qwen relative
coordinates. Earlier Queueboard, Stockroom and ClinicGrid results
are previously scored first-party research and cannot constitute
unseen OOD or legal official release datasets. H287's private FREE T4
LoRA BASE/ADAPTED comparison is already V1 QUEUED on Kaggle and
MUST NOT be re-pushed or reconfigured.

H289 adds an actually different author-owned GUI for a future
source-diverse **DEV research** comparison, without modifying the
already submitted H287 experiment.

## Original synthetic source and disjointness

`h289/routeboard.html` is a hand-authored, single-file, offline
960×720 route-dispatch row interface: six entirely fictional route
records with R-prefixed IDs, English/Arabic RTL, three row
orders, priority filtering, light/night mode and source-owned
per-record action state. No patients, addresses, customers, outside
media, APIs or real data. The Git blob is pinned to
`6b7486384c935ff6ea856a0eefeb213a9504fdf2` and exact
cross-OS source bytes are preserved through `.gitattributes`.

`h289/routeboard_dev_research.py` preregisters twelve synthetic
DEV-only source scenes (six EN and six Arabic RTL) with a single
visible fictional target each, including priority-filtered scenes.
One screenshot and exact bilingual task are passed to a callback
that may later be replaced with a genuinely pretrained/adapted
image-conditioned model. **Only after prediction** does the
source-owned evaluator read the six original route states, execute
the one native H263 action and check whether exactly the requested
fictional record changed and all five neighboring records stayed
unchanged. The snapshot hash must remain identical before/after
policy computation (no stale GUI). Wrong row, abstention and
screenshot mutation are verified failures.

The role string is fixed as
`DEV_RESEARCH_ONLY_PUBLIC_EXPOSED_NOT_ADMITTED_NOT_BLIND_OOD`.
No function trains on or uploads these images. Model-training roles,
official admission, independent blind OOD and release are rejected.
The evaluator's booleans **do not attest genuine model inference**;
a separate authenticated owner-private model run and closed
report are required. Gold/DOM is never passed to a future neural
policy, though the fixed test-fixture callback deliberately uses DOM
to demonstrate the source scorer. That fixture is not Qwen.

## Real qualification

```powershell
python -m unittest discover -s tests -p test_h289_original_dev_research.py -v
$env:HALA_H289_BROWSER_QUALIFY='1'
$env:HALA_H289_BROWSER_CHANNEL='msedge'
python -m unittest discover -s tests -p test_h289_original_dev_research.py -v
python -m compileall -q h289
```

The standalone source-only Github Actions workflow tests on actual
Ubuntu Chromium and Windows Microsoft Edge with pinned versions.
All twelve original scenes are tested with fixture actions; tests
separately verify abstain, wrong neighboring click (executed but
FAILED actual task), and screenshot mutation (refused prior to
native dispatch). No pretrained Qwen, GPU, LoRA, weights or private
media is invoked or published.

## Qualification limits

H289 is a new **public research DEV source family**, not an
independent blind source that was kept secret from the training
team, and cannot be promoted to source-disjoint blind OOD by changing
a label. The H199 admitted TRAIN set stays empty, H173 consumed
OOD remains sealed, H212 old scored procedural tasks barred,
H250/H251 third-party media quarantined. A scientifically
defensible release requires separately rights-qualified lawful
TRAIN/DEV, a new independently controlled unseen private blind
OOD, a properly paired true BASE vs ADAPTED native GUI task-state
improvement and safety/abstention across source families.

Keep all branches stacked DRAFT. Do not merge, deploy, release,
submit paid GPU, expose model weights or publish screenshots,
inference strings or reference coordinates.
