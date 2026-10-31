# H291 — fix the REAL original two-source failure in H287/H290 LoRA

**Verified root cause.** Owner-private FREE T4 H287 V1 ended
`ERROR` with no allowed aggregate. The isolated H290 V2 then ended
`ERROR`, but wrote the exact private whitelisted 546-byte
diagnostic. Its independent H290 schema verifier ACCEPTED a
`SOURCE`-stage `ValueError`, with **zero verified optimizer steps
and no verified model score**. The raw Kaggle logs, screenshots,
predictions and model weights were never read or published.

Independent GitHub source inspection FOUND the fault:
`h285/shiftdesk.html` (Git blob
`f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac`) lives in
H286 at immutable commit
`76a4ebb204d825b280fd4d332ad75591c5f13007`,
but the original Queueboard `h275/queueboard.html` and its
`h276/queueboard_native_research.py` scorer are **ABSENT**
from that branch. They DO exist at different immutable H278
commit `c61409e0b0178227180f0d8e05c67f5a210d01f9`;
Queueboard HTML Git blob
`3cf452e84a75b017905bc14b04a8571411929891`.

Neither V1 nor V2 could have reached real Qwen training. No
experiment gain or training success may be inferred from their errors.

## Narrow source-identity repair

`h291/private_dual_source_lora.py` preserves the frozen H287
8-step balanced EN/AR original ShiftDesk rank4 FP16 relative-aligned
LoRA and the two already-exposed Queueboard research EVAL cases.
Only source resolution and transparent failure telemetry change:

1. Clone **TRAIN** H286 exact branch and commit into ephemeral
   owner-private `/tmp/.../source/train_h286`. Verify clean
   Git HEAD and source SHA1 blob.
2. Clone **EVAL** H278 exact branch and commit separately into
   `/tmp/.../source/eval_h278`. Verify clean HEAD, exact Queueboard
   HTML Git blob and actual `h276` scorer presence.
3. Put both exact source checkouts on the private Python import path:
   H286 for ShiftDesk/H273/H279/H282 and H278 for Queueboard
   original screenshot-locked H276 oracle.
4. Explicit Git `core.autocrlf=false` preserves exact HTML
   bytes on Windows/Linux; independent pinned identities prevent
   silently mixing unrelated source revisions.
5. Failure stage distinguishes TRAIN source from
   `EVAL_SOURCE`. No raw exception messages, gold or model text;
   only a hardcoded categorical private report.

This is a corrected **new owner-private code-only V3** kernel with
a distinct slug
`abdulazizshehri/hala-h291-private-shiftdesk-dualsource-lora-t4`.
It MUST NOT overwrite the failed original H287 V1 or H290 V2.
No original author's source files are changed, copied into public
TRAIN, or attached to Kaggle uploads. Original fixed first-party
research is NOT independent blind OOD.

## Actual independent Windows source validation

In addition to six invented adversarial CPU tests and Python
compileall, an owner Windows ephemeral test executed
`get_pinned_source()` with TWO **genuine separate Git clones**.
It checked exact H286 and H278 revisions, original HTML Git blobs,
the actual source-owned H276 evaluator's import and the H286
training candidate module's import. It succeeded; the two private
temporary Git checkouts auto-deleted with verified removal. This
fixes a REAL terminal GPU blocker, not merely a synthetic test.

The new code-only private package must include exactly one
Python script and one `kernel-metadata.json`, `is_private=true`,
`NvidiaTeslaT4`, no added media, datasets, pretrained weights,
adapters or labels. Remote `/tmp` contains all first-party
rendered PNGs, Qwen weights and optimizer state, deleted on exit.

On a successful real GPU V3 run, only the exact original
H287 bounded aggregate may be read via independent strict H288.
On error, read only
`hala-h291-private-lora-stage-diagnostic.json` through H291's
closed schema verifier. No unfiltered Kaggle outputs, raw logs or
private strings. An `ERROR` with no valid report remains UNKNOWN,
not 0/2 or an inferred optimizer result.

## Scientific limits unchanged

H199 officially admitted TRAIN manifest EMPTY, H173 consumed
old scored OOD sealed, H212 scored prohibited, H250/H251
third-party media quarantined. ShiftDesk and Queueboard are
publicly exposed first-party TRAIN-candidate vs previously scored
EVAL research, **not official training/DEV admission, blind OOD or
generalizable gain**. If an adapter eventually improves on these
two exposed research cases, new independent unseen source-diverse
blind evaluation, rights, abstention and reproducible gains
remain essential before release.

Everything remains DRAFT and unmerged. No publish/deploy/release,
no paid compute and no Mac.
