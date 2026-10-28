# H173 — private owner one-shot evaluation and safe continuation

CURRENT STATUS: SOURCE-CODE PREPARED, NOT MODEL OOD-QUALIFIED.

H172 was already consumed. Frozen H171 visual adapter SHA-256:
8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e.
Frozen H164 language adapter SHA-256:
43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88.

NEVER retry the first rejected H173 owner source at
D:\Hala-H173-Private\ood-32. Its Tingle case 173005 had 6,149
changed RGB pixels on independent replay; the entire first source is
QUARANTINED. Never use it for training, tuning or OOD scoring.

## Stage 1: source-only checks

GitHub Actions has an isolated public-donor Linux browser job, which
already passed 32 native original app cases and all exact RGBA
independent replays. An independent Windows runner job is being
qualified. Neither job reads owner-private prior TRAIN, DEV or OOD
images, nor executes H164/H171 models, so neither is the definitive
owner OOD result.

When authorized Windows device access becomes available WITHOUT
bypassing its quota, update the H173 Draft worktree from the latest
GitHub branch and verify the real original pinned MIT donor trees
remain untouched. Create a completely NEW source directory such as
D:\Hala-H173-Private\ood-32-v2. Do not overwrite the rejected
ood-32. New source must be captured with the updated
h173.capture_ood --mode capture, then verified with --mode verify
using the same seven --owner / --upstream / --h168 / --h169 /
--h171 / --h172 / --dest arguments.

Known owner paths:
- owner: C:\Users\Shehr\work\Hala
- upstream: D:\Hala-H173-Private\upstream
- h168: D:\Hala-H168-Private\train-pilot-32
- h169: D:\Hala-H169-Private\new-dev-32
- h171: D:\Hala-H171-Private\click-train-32-v3
- h172: D:\Hala-H172-Private\new-dev32-v2
- dest: D:\Hala-H173-Private\ood-32-v2 (NEW ONLY)

Capture must pass 32/32 true original-app native reference actions:
16 Tingle close and 16 Tiny Slider correct carousel navigation,
16 Arabic / 16 English, unique raw PNG SHA-256 AND unique decoded
RGBA visual hashes. Sources are the pinned original upstream HTML,
not modified or synthesized UI. Screenshots are 960x720. The pointer
must be neutralized AFTER scrolling the action target; consecutive
images must agree on exact decoded RGBA pixels. A separate completely
new replay checks all 32/32 source snapshots and produces the
private source-replay receipt. Any failure quarantines the ENTIRE
new folder. Never relax the gate, select "good-looking" frames or
reuse any consumed OOD/DEV benchmark.

## Stage 2: private gold-free source and frozen code

Only AFTER 32/32 independent replay, create the owner-private ZIP
using h173.pack_ood --source NEW_DIR --output NEW_PRIVATE_ZIP.
The packing gate checks the source manifest SHA, exact replay
receipt, all unique decoded visual hashes, upstream original source,
and exclusion of owner gold. The ZIP contains ONLY manifest.json and
32 original PNGs. The owner-only reference positions and exact
carousel expected AFTER indexes are never uploaded with model input.

Pin NEW_PRIVATE_ZIP SHA-256 in h173.infer_gpu OOD_ZIP_SHA. It is
deliberately UNQUALIFIED now. Verify local/hosted CI and commit
the new ZIP SHA on a stacked Draft branch. Pin the exact resulting
immutable source commit in h173.kaggle_private_paired_ood SHA and
h173.audit_gpu SOURCE_SHA, and commit again; never use a mutable
branch in the private inference job.

## Stage 3: private FREE T4 frozen paired inference

Prepare a NEW private Kaggle source dataset with only the H173
gold-free ZIP under filename h173-ood.pkg. Mount only this NEW OOD,
frozen H164 language dataset, and a private H171 vision-only dataset.
Never mount H172 consumed DEV or prior H164 OOD.

Run exact 32 BEFORE H164 frozen language-only and 32 AFTER H164
language+H171 vision adapter model predictions on a private FREE
Tesla T4. Zero optimizer steps, zero owner gold, no paid compute,
max 13 GiB CUDA allocated. H173 runner checks immutable SHAs,
dataset membership, frozen model IDs, active adapters, and device.
Independent h173.audit_gpu verifies real 64 proposal receipts and
makes NO claim of native browser task success.

## Stage 4: one-shot original native reward and permanent lock

Only after GPU audit, call h173.native_reward on the NEW owner source,
NEW pinned OOD ZIP, real GPU predictions and pristine upstream. Before
any model action, it writes the OOD consumption lock; interruption
consumes the source and forbids any rerun on that benchmark.

Every case MUST reproduce original 960x720 preaction pixels. Only
the model-generated single pixel click is executed. Correct native
Tingle close success is modal not visible. Correct Tiny Slider
direction is scored using the owner's exact expected AFTER carousel
index; merely moving any slide is NOT sufficient. Owner native gold
is used only by the independent reward scorer, never by the model.

After 64 real browser actions, finalize private OOD receipt and lock.
h173.audit_native independently recomputes totals, Arabic/English,
per-app wins/regressions, paired exact McNemar and the final output
digest. Do NOT publish owner data, model weights or private captures.
Never rescore OOD after it is consumed. Negative or mixed results
are valid experimental results; do not hide them.

## Release and scientific limits

H172 same-app native DEV was H164 4/32 vs H171 15/32 with
13 paired wins and 2 regressions, p=0.007385. It remains
CONSUMED and cannot be used to select any model again. H173 is
genuinely different original apps and must be evaluated only ONCE.
Even positive H173 novel-app OOD evidence is still a tiny 32-case
research result, NOT a release of Hala flagship 35B, production
computer use, or general safety. Keep Draft, protected main
unchanged, no paid tools, no publishing.
