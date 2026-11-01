# H331 — Optional screenshot-only RTL visual-target prompt (unmeasured)

**Status: CODE QUALIFICATION ONLY, not a Qwen improvement or EGM-reviewed design.**

H330 introduced a source-grounded Google Elephant–Goldfish Model (EGM)
design proposal, still **unreviewed by an actual isolated Goldfish**.
This H331 additive, disabled-by-default, zero-cost experiment candidate
tests alternative A from that design without changing any model weights,
training rights, or the existing original-source/oracle evaluation.

## Falsifiable hypothesis

In the genuine, owner-private, already-exposed H324 pretrained Qwen
BeaconBoard pilot, English succeeded on one case, Arabic RTL failed
`wrong_focus_or_state` after a single generated click. One case cannot
establish language-level accuracy or the root cause. H326 verified that
**correct** source-owned button centers are correctly converted by the
existing H273 0–999 normalized point mapping in 6 English and 6 Arabic
RTL exposed scenes, ruling out a simple deterministic x-axis inversion.
A plausible but unproven alternative is that the model failed to
identify which visible station button matched the user instruction.

The candidate, `visual_target_rtl`, adds fixed screenshot-only
instructions to compare the task's station identifier with the text
actually visible on each button, choose the *matched button's* center,
treat RTL as text direction rather than mirrored image x coordinates,
and abstain when visually unsure. It supplies **no source DOM, gold,
target coordinates, reference labels, hidden state, or source oracle**.

## Implementation and controls

- `control`: original H323 `model_visible_prompt`, byte-identical
  for both stages; unchanged default behavior and default bridge
  three-positional-argument policy construction.
- `visual_target_rtl`: fixed added text **only on STAGE_CLICK**;
  STAGE_TYPE is byte-identical to the control. Native JSON grammar,
  H273 coordinate conversion, source oracle and strict native H263
  dispatch remain exactly unchanged.
- All variant names are in a closed allowlist. Invalid variants or
  stages fail closed before any source dispatch.
- Opt in only with
  `run_preloaded_dynamic_source_research(..., prompt_variant="visual_target_rtl")`
  in a separately authorized model-time research caller; neither
  `h324/private_pretrained_beaconboard_two_case.py` (pinned immutable
  original source) nor any prior H324 aggregate is modified. Model
  identity/weights/free T4/quota/privacy must be independently verified
  by the actual caller, not inferred from this code.

Run offline invented-only tests:

```sh
python -m unittest discover -s tests -p test_h331_optin_visual_target_prompt.py -v
python -m unittest discover -s tests -p test_h323_dynamic_preloaded_bridge.py -v
```

These tests check fixed control parity, stage isolation, rejection of
unsafe variants, absence of hardcoded source answers in the prompt
delta, and fake preloaded bridge plumbing. **Mocks cannot demonstrate
image understanding, relative click accuracy or generalization.**

## Prospective research rule (NOT yet run or authorized)

A real follow-up must preregister frozen, original source-disjoint DEV
tasks; pinned base Qwen model revision, identical sampling and native
action policy; true paired `control` versus
`visual_target_rtl` source task-state success, first-click/focus,
abstention, malformed output and errors; complete denominators;
and an uncertainty criterion. Both variants must receive the same
task/screenshots with no reference leak. A matched public-exposed
BeaconBoard smoke would only be **debug research**, never fresh blind
OOD or evidence that a trained Hala adapter improved. Actual gain
remains unproven; even a prospective prompt win is *not* LoRA learning.

Keep the H199 admitted TRAIN set empty; H173 scored blind OOD sealed,
H212 barred, H250/H251 rights assets quarantined, H293/H294 private
access safety barriers intact. No GPU, paid service, model training,
new EGM Goldfish attestation, private media publication, merge,
deployment, public release, or Hala-ready assertion in H331.
