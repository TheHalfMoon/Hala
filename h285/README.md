# H285 — original bilingual ShiftDesk TRAIN-candidate factory

**This is a genuinely new, author-written GUI source family for
candidate training data, not a fourth scored original model experiment,
not a DEV/OOD benchmark, and NOT yet an officially admitted release.**
It does not read any previous private/consumed OOD data or user files.

## Why this step

H279 Stockroom pretrained research measured 1/2 source-state success;
H279 Queueboard corrected V3 measured 1/2; H282 ClinicGrid pretrained
research measured 2/2 on two previously exposed source-owned tasks.
All three sources have already been exposed/scored and cannot prove
adapted Hala generalization or serve as independent unseen blind OOD.
Previous H270 LoRA attempts on Stockroom also do not establish robust
paired model improvement.

H285 changes direction: **manufacture future training supervision from
a new, disjoint original GUI family**, before any new adapted-model
evaluation. The final release gate at
`h199/native_action_sft.py:ADMITTED_TRAIN_MANIFEST_SHAS` intentionally
stays EMPTY. H285 is not a shortcut into that gate.

## Original author-owned ShiftDesk

`h285/shiftdesk.html` is a single-file 960×720 offline application
with eight **fictional** queue items. It has no patient, personal,
enterprise, external media, third-party assets or network source.
It supports Arabic RTL/English, three distinct card orders, two
light/dark themes and two card densities. The original HTML Git blob is
pinned at
`f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac`, with cross-OS
`.gitattributes` byte preservation.

`h285/train_only_browser_candidates.py` deterministically chooses
**24** different source-view combinations: 12 English, 12 Arabic.
Each prompt names the exact fictional visible item for one
`Complete`/`إنجاز` click. A real offline browser screenshot becomes
the prospective training image. **Only the TRAIN candidate extractor**,
never a model or evaluator, obtains the original HTML button bounding
box and produces one strict H199 supervised native screenshot-pixel
click. This is a *training label*, not a model-predicted action.

The extractor then uses H263's real native click actuator and verifies
that exactly the requested item changed status, all seven other
source IDs remained unchanged, and the visible counter advanced to 1.
It requires all 24 actual actions to work, all 24 screenshot PNG
byte-hashes to differ and all 24 fully decoded RGBA pixel-hashes to
differ. If any source drift, duplicate image, target mismatch or
task-state failure occurs, **no candidate is sent to the sink**.

After the entire set passes, an explicitly trusted **in-memory**
training callback may receive `(png_bytes, source_task_prompt,
native_supervision_json)`. The tool itself never saves raw images,
reference points, or training completions to public GitHub, logs or
Kaggle output. The returned summary contains only source role,
count/locale numbers and explicit unqualified status flags.

## Qualification

Source-only tests:

```powershell
python -m unittest discover -s tests -p test_h285_original_train_only_browser.py -v
```

Real local owner **Windows Edge**, with opt-in:

```powershell
$env:HALA_H285_BROWSER_QUALIFY='1'
$env:HALA_H285_BROWSER_CHANNEL='msedge'
python -m unittest discover -s tests -p test_h285_original_train_only_browser.py -v
```

The dedicated free GitHub Actions source/browser workflow checks
actual Windows Edge plus Ubuntu Chromium (Python 3.11; pinned public
Playwright and Pillow). No GPU or model weights are needed to
qualify the supervised source sampler.

## Independent review / admission still required

This is currently **TRAIN_CANDIDATE_ONLY_NOT_OFFICIALLY_ADMITTED**.
Authorship and source history should be reviewed independently,
candidate role permanently segregated from all preexisting original
research scores, and decoded pixel disjointness against all *legally
available* historical scored/heldout sources proven *without opening*
sealed or consumed OOD gold. No helper may simply mark the manifest
officially admitted, alter H199, or claim this source's public GUI
as independent blind OOD.

A future private zero-cost Qwen training run must keep rendered frames,
labels, adapters and model weights only in ephemeral authorized
noncommercial Kaggle storage and emit only a bounded sanitized
aggregate. Training then requires truly new source-disjoint DEV,
paired BASE/ADAPTED genuine browser actions and a separately
preserved never-exposed one-shot blind OOD. This PR **does not**
authorize releasing Hala or merging to main.
