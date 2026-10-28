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

**Owner Windows 2026-10-29:** The separate `ood-32-v2` attempt captured 32
original scenes but failed its one-and-only independent replay at Tingle
`173006`. Its replay-attempt lock is consumed: quarantine the ENTIRE v2,
never retry, rescore, repack, or upload any v2 artifacts. Public-source-only
cross-Chromium-process QA reproduced intermittent exact RGBA drift using
untouched pinned MIT apps on the same owner Windows host. A pinned native
software rasterizer (`--disable-gpu`) passed one full 32/32 cross-process
PUBLIC-source run, but this alone is not new owner OOD qualification.
Those prerequisites subsequently PASSED at Git source SHA
`692d9475ebfbf57a696088fe269173664d75ddae`: 4/4 hosted
Python/Linux/Windows jobs and three independent owner-Windows PUBLIC
cross-process 32/32 replay runs. A completely NEW owner `ood-32-v3`
then PASSED 32/32 source-native capture and the sole independent exact
RGBA replay. The private gold-free archive contains precisely 32 original
PNGs plus manifest.json; owner reference actions remain outside the ZIP.
Frozen v3 ZIP SHA-256:
`9775b2bfaad88a29e3ce4e1722ba86f417ab579c1d1ab1b8de9721e235266c18`.
Owner v3 `capture`, `verify` and `pack` stages are now DONE AND CONSUMED.
NEVER retry, rescore, recapture or repack v3. v1 and v2 remain quarantined.
**Still NO model OOD inference, no native model reward, no OOD quality claim.**
The next and only eligible step is immutable final source pin, private
zero-cost frozen T4 inference, independent GPU audit, then the separate
one-shot native reward, with exact source/proposal identity and no old data.

## Stage 1: source-only checks

GitHub Actions has an isolated public-donor Linux browser job, which
already passed 32 native original app cases and all exact RGBA
independent replays. An independent Windows runner job is being
qualified. Neither job reads owner-private prior TRAIN, DEV or OOD
images, nor executes H164/H171 models, so neither is the definitive
owner OOD result.

HISTORICAL v3 capture steps below are retained only as provenance.
**DO NOT EXECUTE THEM AGAIN: v3 is already captured/replayed/packed.**
When authorized Windows device access becomes available WITHOUT
bypassing its quota, update the H173 Draft worktree from the latest
GitHub branch and verify the real original pinned MIT donor trees
remain untouched. Create a completely NEW source directory such as
D:\Hala-H173-Private\ood-32-v3. Do not overwrite the rejected
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
- dest: D:\Hala-H173-Private\ood-32-v3 (NEW ONLY)

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
the new ZIP SHA on a stacked Draft branch. Freeze the **final actual
40-hex Git commit SHA AFTER that commit completes**. DO NOT embed this
source SHA into the files that form that same commit: such a self-pinned
SHA would change the commit and can never match. Instead, supply it
through the private owner/Kaggle environment variable:
`HALA_H173_PINNED_SOURCE_SHA=<final_immutable_commit_sha>`.
The standalone Kaggle runner checks out exactly this SHA before
importing H173, then records it in the private job receipt. Independently
audit with `python -m h173.audit_gpu --output PRIVATE_GPU_DIR
--expected-source-sha <same_final_immutable_commit_sha>` (or set
the same private environment variable before native scoring). The
auditor refuses a missing/mutable/untrusted source identity, even
when a job record claims a valid commit. NEVER use a mutable branch.

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
