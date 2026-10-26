# H127 — blind original TRAIN64 visual identifiability screening

**Research-only and owner-private. NOT model training, new benchmark results, or
model release qualification. No human review has yet been performed.**

H126 verified exactly 64 new source-unique original H71 TRAIN screenshots
(16 each sheet, slide resize, slider and text highlighting), with SHA-qualified
exclusion of 2,624 previously used/withheld original sources. H126 was strictly
a lexical/source-geometry analysis. It did NOT establish whether the authentic
task instruction and pixels make either START or END visually identifiable.
H125's trained-independent ShowUI-2B / UGround-V1-2B original TRAIN64 baselines
were both **0/64 strict full drag**. H127 tests whether these tasks are
visually specified at all *before* another costly adapter/training run.

## Owner-private preparation

1. Freeze this H127 source commit and run only the owner-private,
   noncommercial **CPU** Kaggle kernel declared in `kernel-metadata.json`.
   It reconstructs genuine original H71 TRAIN media with prior H126
   admission code and SHA verification, re-derives the frozen 64 selection,
   and outputs a **private** blind ZIP plus a **separate PRIVATE ground-truth
   JSON**. Both files are sensitive; never add them to Git, public Kaggle,
   a PR comment, or any public data endpoint.
2. Give a fresh copy of **only** the ZIP to each of two independent human
   reviewers. Keep the private truth JSON out of their reach, and do not
   show earlier model predictions or the other reviewer's annotations.
3. Build the viewer separately for each reviewer in a new private folder:

   `python -m h127.score_cli viewer --pack blind-review-pack.zip --output private-reviewer-A`

   Open `private-reviewer-A/index.html` locally in a browser. The offline
   page uses no libraries, external requests or servers. It displays the
   original instruction and screenshot, lets the reviewer choose native-pixel
   START and END points, or mark ambiguous/unreadable. On completion of all
   64, export the review JSON. Repeat independently with reviewer B.
4. Once two independently gathered reviews exist, the owner alone runs:

   `python -m h127.score_cli score --truth PRIVATE-truth-do-not-share.json --review-a A.json --review-b B.json --output PRIVATE-H127-aggregate.json`

   The output includes per-domain/overall counts and omits individual
   instructions, pixels, coordinates, gold boxes, or source IDs.

## Preregistered *screening* rule, not model qualification

- Exactly two distinct **self-attested** blind reviewer IDs and 64 responses each.
- A full drag counts only when both manually selected START and END pixels are
  **inside their respective authentic original source bounding boxes**.
- Both reviewers must jointly identify >=58/64 full drags and >=13/16 in
  **each** domain, with <=4 abstentions per reviewer.
- Passing produces `PROVISIONAL_SELF_ATTESTED_SCREEN_PASS`. There is NO
  automated verification of actual reviewer independence or identity; that
  requires real external review arrangements and documentation.
- Any failure remains `FAIL_OR_AMBIGUOUS_SOURCE_IDENTIFIABILITY`.
  If visually ambiguous, repair data/task definitions with **new original TRAIN**
  sources before adapting models. Do not relax old EVAL/TEST thresholds, or
  recycle any prior heldout images into training.
- No inference, optimizer steps, EVAL/TEST usage, model weights, paid compute,
  merge, release, or deployment in H127.

Synthetic local verification:
`python -m unittest discover -s tests -p "test_h127_blind_gui_gate.py" -v`

The blind ZIP is deliberately private, and the browser session is private,
not a hosted web service. Work on reviewer identity verification, blind
annotation and qualified model training remains open.
