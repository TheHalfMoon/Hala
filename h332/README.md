# H332 — frozen, matched exposed-source prompt study protocol

**Status: prospective infrastructure; no new GPU or genuine Qwen trial.**
H331 was qualified by GitHub CI 26/26, but was not shown to improve
model accuracy. H332 prepares a controlled comparison, following the
H330 Google EGM design proposal and its documented gaps. It does not
pretend that an isolated Goldfish review took place.

## Scientific question

The previously observed H324 genuinely pretrained Qwen3-VL-2B result
was 1/2 on PUBLICLY EXPOSED first-party BeaconBoard; EN task passed,
AR RTL task failed after wrong initial focus/state. One case for each
language cannot estimate model generalization. H326 showed that
pre-supplied correct target centers work through H273 RTL coordinate
conversion, but did not prove Qwen recognizes targets visually.

H332 compares the existing H323 `control` prompt with H331's opt-in
`visual_target_rtl` first-click visual-target instruction, using
**the exact same** model, original source scenario and deterministic
generation/runtime. This is *exploratory source-exposed development
research*, not independent blind OOD.

## Frozen protocol

- Scene indices: `(0, 2, 6, 8)` from H319: exactly 2 English and
  2 Arabic RTL original-public source GUI cases, selected and frozen
  in code. This small number is **NOT** a statistical performance
  sample or preregistered release-level model evaluation.
- Each scene runs both prompt arms in fresh browser state, not the
  second arm in the state left by the first. Arm order is
  counterbalanced two `control → candidate`, two
  `candidate → control`.
- A source-only preflight renders the *same scene twice* in isolated
  offline browsers, demands the same 960×720 decoded RGB first image
  and absence of any open station before inference. The source oracle
  never gives button positions, DOM or code targets to a policy.
- All real inference, when independently authorized, must call the
  existing H323 preloaded screenshot-only Qwen bridge, preserving
  token-level native JSON grammar, H273 0–999-to-pixel conversion and
  H263 real native browser action. The arm can differ ONLY in first
  click's fixed prompt text. Stage-two type prompt and policy stay
  identical. Failure/refusal is a counted outcome, never discarded.
- Per-scene outcomes stay in RAM. The strict source wrapper checks
  H323 categorical failure shapes and demands all eight arm runs.
  Export only per-language/arm totals for task success, native actions,
  generations, bounded failures, and per-language pair win/loss/ties.
  No per-case model predictions, labels, screenshot/media bytes, RGB
  digests, raw prompts, target IDs, actions, task gold, point
  coordinates, private model output or weights are returned/persisted.
- An injected mock runner exists **ONLY** for CPU contract testing;
  neither its success nor the study protocol can independently attest
  Qwen identity. A future owner-private free T4 caller must independently
  attest exact model revision and submitted code, source pins,
  quota, privacy and stage-safe aggregate after real model generations.

The result has `source_public_exposed_not_blind=True`,
`optimizer_steps=0`, `adapted_model_gain_proven=False`, and
`can_release=False`. Even if the prompt candidate wins on all four
scenes, **that is not a learned LoRA improvement**. Never recycle the
scenes into hidden OOD. Training H199 remains empty.

## Validation

Source-only CPU tests check the complete hypothetical paired receipt,
impossible protocol transitions, injected private points and false
blind/LoRA/identity claims, nonmatching arm counts and four frozen
scene orders. With `HALA_H332_BROWSER_QUALIFY=1`, actual original
Chromium or Edge opens each frozen source scene twice and compares the
first decoded RGB frames. CI uses the complete repo for both.

A future Qwen GPU submission is a **separate decision**: first verify
H332 exact-head CI, no concurrent/paid GPU, model/source license, owner
private kernel quota, safety and source rights; use ONLY existing
non-sensitive public/exposed source. Do not attempt a private output
retrieval workaround or bypass blocked H293/H294. This PR neither
creates a Kaggle kernel nor requests inference. The follow-up must
explicitly distinguish measured prompt tuning from training or new
blind OOD and preserve full task denominators.

Keep all files in a stacked Draft, do not merge/release/deploy,
reopen H173 sealed OOD, admit H212, reuse H250/H251 rights-quarantined
assets or bypass H293/H294 safety constraints.
