# H296 — FieldPort: original bilingual two-action form-use research

**A new author-owned, fully synthetic, PUBLIC/EXPOSED original offline
GUI for screenshot-only computer-use sequence evaluation. This is NOT
Qwen model inference, a newly admitted TRAIN/DEV source or blind OOD.**

## Why this is different from Queueboard and RouteBoard

Hala's validated original GUI research through H295 predominantly
measured one-step **click** actions on queue cards. Yet H199/H263
already define a true two-step interaction: **click to focus a visible
input**, then **type_text** into that *already focused* field.
The action parser strictly forbids combining those actions in one
prediction. Previous first-party queue/button cases do not prove
this sequence works in an independent UI family or either language.

H296 adds **FieldPort**, a genuinely author-owned, single-file,
960×720, offline fictional service-code form. It uses **four
fictional input fields** with a different layout from all existing
queue/route app sources, three row orders, light/night themes, and
Arabic RTL/English LTR. Twelve frozen original DEV **research**
scenes (6 EN + 6 AR) each request entering one harmless fictional
`FIC-xx` code into one specific `F-xx` field. No real names,
patients, addresses, accounts, services, external media, APIs,
datasets or third-party content. HTML exact source Git blob:
`52135ef3f3a68922b82be7bcb95a337a89982b18`
preserved across OS via `.gitattributes`.

## Strict real native two-screenshot source evaluator

`h296/fieldport_sequence_research.py` uses unchanged H199 strict
single-native-action JSON, H263 actual local browser actuator,
H264 screen capture and screenshot lock, and a fixed disconnected
browser context that blocks external network and downloads.

For each original, separately configured, fresh synthetic scene:

1. Capture **real screenshot #1**. Provide ONLY its PNG bytes and
   source-user-visible bilingual task to the policy callback.
   Verify the screenshot and page URL did NOT change during the
   callback. Strictly parse one H199 action; execute ONLY a native
   click or abstain. Check all four original field values remain
   initially empty (focusing must not mutate form data).
2. If clicked, capture **real screenshot #2** after focus. Again
   send screenshot/task ONLY to the callback, verify exact snapshot
   freshness, parse one action, dispatch a native H263
   `type_text` into the *currently focused editable field*, or
   allow abstention.
3. **Only after the two model-facing actions**, source-owned
   `fieldPortState()` checks ALL FOUR field values. Success is
   strictly one exact requested fictional code in the requested
   field and the other three fields unchanged. A click on the
   wrong field followed by a successful type call is a task
   failure, not a false success. Wrong code and abstain fail too.
   A `type_text` before focus fails closed at H263 without
   fabricating a completed task.

The original DOM reference/field locations are used by the
**source-owned scorer and test fixture ONLY** after/beside
action dispatch. They must NEVER enter a neural policy or model
prompt. H296's tests use **fixed non-neural fixtures** deliberately:
this exercises genuine browser execution but is **not Qwen
model-time inference** or new learned skill.

## Exact source and qualification

```powershell
python -m unittest discover -s tests -p test_h296_form_sequence_research.py -v
$env:HALA_H296_BROWSER_QUALIFY='1'
$env:HALA_H296_BROWSER_CHANNEL='msedge'
python -m unittest discover -s tests -p test_h296_form_sequence_research.py -v
python -m compileall -q h296
```

With explicit browser opt-in, real Windows Microsoft Edge qualifies:
- all **12 original bilingual two-step tasks** with 24 actual
  native browser focus+type dispatch calls;
- all four post-action source field state records;
- unique first-screen snapshots across registered scenes;
- wrong field, wrong fictional code, abstain at first/second
  step, screenshot mutation, unsupported first action and
  type without focus negatives;
- source Git blob, no role promotion, RTL and no network.

A standalone hosted CI workflow runs the same cases using **real
Ubuntu Chromium AND Windows Edge**, independent of the owner PC,
and never downloads models or uses GPU, Kaggle, paid compute or
private human/source data.

## Source exposure and finish conditions

H296 is **publicly exposed DEV research** because code and task
fixtures are in GitHub. Calling it new blind OOD would be scientifically
false. It is not added to official H199 TRAIN admission; it does not
resolve H173 consumed blind OOD, H212 scored tasks or H250/H251
external third-party quarantine. It is a genuine additional action
modality/source qualification rather than evidence that a trained
model improved.

Hala still needs independent rights/lineage-qualified diverse
TRAIN/DEV, a brand-new never-exposed independently held-out blind
OOD, real BASE-vs-ADAPTED multisource native-state improvement
and abstention/safety before any model release. All PRs remain
stacked DRAFT; no main merge, deployment, paid model inference
or publishing of model weights/media.
