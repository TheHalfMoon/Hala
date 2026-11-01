# H323 — Source-only preloaded Qwen bridge for BeaconBoard

**Code wiring only. This stage contains zero real pretrained generations,
zero GPU usage, zero legal data admission and zero model-skill scores.**

H323 adds a genuinely different model-time grammar for the NEW public
original dynamic BeaconBoard app. The H297/H298 FieldPort prompt expected a
pre-existing text INPUT and `FIC-NN` code, so it MUST NOT be reused unchanged
for BeaconBoard's STATION BUTTON and `SIM-NN` task. H323's frozen grammar
clicks a screenshot-grounded button with normalized 0..999 coordinates,
then after a genuine native button click reveals a new focused input,
generates a second screenshot-conditioned SIM-NN code action. The token
filter is installed **before** each actual model.generate, not a posthoc
text repair. No DOM, hidden reference target, source HTML or gold point is
passed to the VLM.

The H319/H320 source evaluator alone checks original all-four station
state after real browser action dispatch and ensures that model-visible
decoded 960x720 RGB pixels are new for stage two. The H323 bridge
cross-checks typed/native click counters, model-time token filters,
per-stage distinct screenshot counts, categorical refusals and terminal
safety. Source success without exactly two genuine native dispatches is
refused. The policy seals wrong-focus and wrong-code outcomes. Its
receipt intentionally says independently attested pretrained identity
**false**: a future owner-private runtime must verify pinned
`Qwen/Qwen3-VL-2B-Instruct` model revision, free noncommercial T4,
source commit and actual model.generate calls independently.

Run invented-only CPU contract tests:

```sh
python -m unittest discover -s tests -p test_h323_dynamic_preloaded_bridge.py -v
```

These tests use FAKE action strings and a declared mock policy. They do
not demonstrate vision intelligence. H300 verified genuine pretrained
FieldPort source 12/12 on exposed tasks; that result is not transferable
to BeaconBoard and is not an adapted/LoRA gain. H291/H292 LoRA BASE 6/8
versus ADAPTED 6/8 did not show improvement. H199 official TRAIN admission
remains EMPTY; H173 scored OOD is sealed, H212 barred, H250/H251 rights
quarantined, and H293/H294 safety blocks cannot be bypassed. No merge,
model weights, private screenshots/outputs, paid compute, release or
publication. H323 is prerequisite code, not experiment authorization.
