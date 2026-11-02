# H358 — Original NoteRelay bilingual multi-step TRAIN/DEV candidates

**NO OFFICIAL TRAIN ADMISSION. NO BLIND OOD. NO QWEN TRAINING OR INFERENCE.**

H357 provided a genuinely original, single-action screenshot-click GUI
family. H358 introduces a *different authored application and action
type*: `noterelay.html`, a standalone responsive HTML form workflow
written from scratch. Its actual browser state progresses through
**Open new note → Focus reference-code input → Type synthetic code →
Save note**. English and Arabic source UIs use native LTR/RTL
layout with explicit browser text rendering. The only input strings
are invented codes, e.g., `A1000`; no patient names/records, real
accounts, government services, copyrighted external assets, external
fonts, network calls or secrets are involved.

The factory enumerates 32 original source scenes (16 English /
16 Arabic, 16 light / 16 dark), giving **128 distinct pre-action
screenshots and 128 native logical one-action labels**:
- 96 native screenshot clicks (`open`, `focus`, `save`),
- 32 separate `type_text` actions (after independently observed focus),
- 32 completed original four-step native workflows,
- 32 stable double initial-scene screenshot captures,
- 128 independently checked original browser state transitions.

Synthetic gold/reference points come exclusively from the ORIGINAL
browser used as a separate training-candidate oracle. A future model
would see only a screenshot and the original English or Arabic user
task, not DOM/CSS/element selectors, expected code, source labels,
hidden gold or reference clicks.

`source_candidate.py` is source-only infrastructure; calling
`capture_candidate` explicitly with a caller-owned fresh temporary
folder creates `candidate-rows.jsonl`, `candidate-manifest.json`
and PNGs. Test invocations create these files ONLY under Python
`TemporaryDirectory`, then delete them, including on exceptions.
**Git records only HTML, the Python factory, docs, tests and CI**.
There is no persistent training dataset and no private benchmark.

## Real browser qualification

```sh
HALA_H358_BROWSER_QA=1 python -m unittest tests.test_h358_original_noterelay_multistep -v
```

On Windows, use an installed Edge browser. On Linux, install pinned
Playwright Chromium. The dedicated 4-job hosted CI runs exact original
browser QA under Windows/Ubuntu Python3.11/3.12, plus CPU-only H199 and
H355/H356 rights/independent-OOD fail-closed tests.

The report is explicit:
- `schema=hala.h358.noterelay.original-candidate.v1`;
- `role=ORIGINAL_MULTISTEP_TRAIN_DEV_REVIEW_CANDIDATE_NOT_ADMITTED`;
- `official_train_admitted=false`, `independent_rights_approved=false`;
- `has_source_disjoint_dev=false`, `fresh_blind_ood=false`;
- `actual_model_calls=0`, `gpu_calls=0`;
- `can_train=false`, `can_release=false`.

This differs from H357 source family and covers multi-step focus
and typing, but neither app has an authenticated independent external
rights/title decision for rendered ML screenshots or resulting model
weights. Their authorship alone does not establish all screenshot,
font, browser and asset rights for every intended purpose.

Neither H357 nor H358 is a fresh independent blind test; both are
PUBLIC/EXPOSED source-only candidates in the development repo.
A single additional original app does not establish sufficient
source-family generalization or DEV/TEST independence. There is
NO Qwen learned-weight result.

Human scientific gates remain unchanged: external qualified rights
review [#384](https://github.com/TheHalfMoon/Hala/issues/384),
separately independently custodied unseen family-disjoint EN/AR OOD
[#385](https://github.com/TheHalfMoon/Hala/issues/385), explicit
H199 source-manifest TRAIN admission (currently EMPTY), real
zero-cost legally authorized adapter training, paired BASE versus
ADAPTED independent held-out results and safety/abstention evidence.
Do not merge, deploy or release Hala based on H358.
