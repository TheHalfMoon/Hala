# H298 — actually connect the preloaded VLM call path to two-stage form actions

**Engineering qualification only. There has been NO H298 owner-private
Qwen real inference, no trained model improvement, no official TRAIN/DEV
admission, and no new independent blind OOD.**

## Previously verified concrete behavior

- H296 author-owned PUBLIC/EXPOSED offline bilingual FieldPort form
  qualifies **12 real-browser original source-fixture tasks**, 6EN/6AR,
  each `click-to-focus` then a separately observed `type_text` action
  with ALL four fictional fields checked by source-owned postaction
  state. Dedicated Ubuntu Chromium and Windows Edge CI each 10/10,
  snapshot mutation and wrong focus/code/abstain controls included.
- H297 exact model-time grammar *primitives* are qualified on CPU,
  7/7 normal and optimized Python. The frozen H273 relative 0..999
  click/abstain grammar is separate from a strict synthetic
  `type_text` / abstain second step. No combined model response,
  no gold/example-correct-code grammar shortcut.

Neither the browser fixtures nor grammar primitive tests established
the real MULTIMODAL PROCESSOR/GENERATE wiring between them.

## New H298 preloaded policy integration

`h298/preloaded_form_two_stage_policy.py` connects existing
`H208.PreloadedVisionPolicy`,
`H268.NativeGrammarModel`,
`H279.RelativePrevalidationProcessor`, `H273` relative-to-native
bridge, and **H297 state+token-grammar** to the unmodified H296
original two-screenshot source-owned native sequence scorer:

1. The source H296 callback supplies ONLY an original 960×720
   screenshot PNG and the frozen `source_task_prompt(user_task)`.
   H298 checks that exact prompt identity and rejects stale or
   duplicate screenshots; no DOM source gold/target position enters
   this model-facing policy.
2. On the first call, install the unchanged H273 real
   normalized model-time click-or-abstain token filter on the
   **already loaded** model via `NativeGrammarModel.generate`.
   `PreloadedVisionPolicy` forms real image-encoded Qwen-compatible
   multimodal inputs (not a text-only branch).
   `RelativePrevalidationProcessor.batch_decode` performs
   strict H273 normalized-to-native conversion during decoding,
   advancing H297's trusted state from the original model text,
   before the unchanged strict H199 native parser.
3. After a true click and a **different new screenshot showing
   the focused input**, install a separate H297
   `type_text` / abstain LM-format-enforcer token restriction
   on a second wrapped model-time `generate`.
   The exact native response is strictly checked without
   patching/repair. H263 executes a second action only after
   verifying the editable element is currently focused.
4. The real original H296 scorer independently checks all four
   actual field values. Wrong focus even with a dispatched
   type call is TASK FAIL, as are wrong code and abstention.
   A malformed response, duplicate source screenshot, changed
   source task, missing generation/filter, third model turn or
   inconsistent actual native dispatch fails closed.

No predicted action JSON, raw model text, screenshot, coordinate,
gold/source field reference or model weight is ever saved into
GitHub or written into the H298 safe protocol counters.

## Tests and what they do NOT prove

```powershell
python -m unittest discover -s tests -p test_h298_preloaded_form_two_stage_policy.py -v
python -O -m unittest discover -s tests -p test_h298_preloaded_form_two_stage_policy.py -v
$env:HALA_H298_BROWSER_QUALIFY='1'
$env:HALA_H298_BROWSER_CHANNEL='msedge'
python -m unittest discover -s tests -p test_h298_original_form_policy_browser.py -v
python -m compileall -q h298
```

- **7 CPU tests** use a small fake preloaded model and model-time
  grammar gate, exercising ACTUAL H268 generation wrapper,
  ACTUAL H279 relative prevalidator and H297 state transitions.
  It rejects bad prompts/images, duplicate screenshots,
  type-before-focus, click-in-typing-stage, unauthorized JSON and
  false attestation. They do **NOT** load HF/Qwen or invoke
  a real model-time token filter over the genuine vocabulary.
- **4 genuine source browser integration tests** launch real
  Microsoft Edge/Ubuntu Chromium and actuate the original form
  via the entire H298 policy and H296 source scorer. **Model
  outputs are intentionally STUBBED using a test-only source
  fixture**. Four synthetic EN/AR correct two-action sequences
  plus wrong-field, wrong-code and abstention are exercised.
  These browser results do **NOT** represent independent
  neural model behavior or accuracy. No private model files.
- Dedicated H298 GitHub workflow runs both CPU and browser
  tests on Ubuntu Chromium and Windows Edge. Original H296,
  H285/H289/H281 browser workflows and broad source CI must
  remain green before accepting this Draft PR.

The safe `preloaded_model_contract` summary is deliberately
**UNATTESTED**: `pretrained_model_identity_attested=false`,
`real_owner_private_gpu_run_attested=false`,
`native_source_task_success_attested=false`,
`blind_ood_proven=false`. Even when the source scorer reports
a synthetic task success, its model identity and real inference
status are NOT inferred.

## Scientific and release restrictions

Genuine H291 and H292 owner-private FREE T4 runs each verified
8 changed LoRA optimizer updates but reported **ZERO** native
paired task-state gain: BASE 1/2 vs ADAPTED 1/2 Queueboard,
and BASE 5/6 vs ADAPTED 5/6 RouteBoard in H292.
H293 exploratory 24-step private GPU run terminal ERROR; result
access was blocked by safety checks; no training count/score
or runtime stage is known. **Never bypass/retry that blocked
private result through alternate channels.**

Official H199 admitted TRAIN manifest remains EMPTY, old H173
OOD scores consumed/sealed, H212 scored data prohibited,
H250/H251 external assets still quarantined. All FieldPort
sources are PUBLIC EXPOSED research, never blind OOD.
Further legitimate progress requires separately attested actual
owner-private pretrained Qwen multi-step GUI task outcomes,
rights/lineage-qualified independent source-diverse TRAIN/DEV,
a genuinely new never-exposed independently custodied blind
OOD, real adapted model benefit and safe abstention.

Keep all source changes in stacked DRAFT PRs. No main merges,
deployment, paid GPU, Mac, raw model outputs or model publication.
