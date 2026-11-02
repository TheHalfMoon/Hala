# H361 — H199-native original candidate → Qwen3-VL processor parity

**NOT TRAINING. NOT AN ADMITTED DATASET. NO MODEL WEIGHTS, GPU, MODEL
INFERENCE, BLIND OOD OR RELEASE.**

H359 verified H199's strict native screenshot-action records and
identified that the older logical-DOM `training.artifacts` loader is
incompatible. H361 provides the next necessary bridge: a read-only,
bounded **processor-only** validation path for those original rows.
It does not silently adapt the legacy `{"type":...}` supervision.

## Input scope

The source families are the publicly exposed, self-authored H357
ControlGarden and H358 NoteRelay, not new blind DEV/TEST. It audits
**all 64 + 128 original browser source rows** under H359 before
loading PIL or a Qwen processor. It then selects exactly **12**
frozen, already exposed source examples for processor parity:
4 H357 clicks and 8 H358 actions, 6 English and 6 Arabic
(10 clicks and 2 `type_text` actions). No new screenshots,
model outputs or token sequences are committed or logged. Temporary
screenshots and labels used by tests are destroyed at end of each test.

Each model-visible message contains the *same original screenshot
pixels* and original user task; the assistant answer is exactly the
H199 one-step native action. H361 rejects wrong raw PNG identity,
wrong original Git source SHA, malformed or unadmitted manifest,
non-native action, changed multimodal prompt prefix, different
pixel tensor/image grid, assistant loss covering image/user tokens,
a missing supervised continuation, and decoded completion that
doesn't roundtrip to the original strict H199 action.

## Real pinned Qwen3-VL processor option

`run_pinned_offline_candidate_processor(snapshot, datasets)` is
callable ONLY with both original source candidate roots and an offline
cached Qwen3-VL-2B processor snapshot. It calls H359 source verification
first; then H216/H218 independently verify the processor's exact
published **eight pinned processor assets** and revision
`89644892e4d85e24eaac8bacfd4f463576704203`.
It refuses to fetch tokens or network assets; it requires
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The real
Qwen3VLProcessor class must load locally. No model weights are
requested, no `model.generate`, forward pass, optimizer or adapter.

**Important evidence boundary:** the owner Windows PC presently lacks
`torch` and `transformers`. H361 real pinned Qwen processor
parity has **NOT BEEN EXECUTED OR QUALIFIED**. Its CPU-source tests
run actual Edge/Chromium images but a deliberately named **FAKE
processor** for interface-contract negative tests. Those tests
must never be reported as a real tokenizer/processor PASS. An
independent resource-qualified zero-cost environment and exact
pinned processor snapshot are needed for subsequent real parity.

## Engineering qualification

```sh
HALA_H361_BROWSER_QA=1 python -m unittest tests.test_h361_original_qwen_processor -v
```

The owner Windows Edge source and fake processor interface completed
two tests including 12 source sample interactions, checks on
192 original PNG/row identities via H359, and adversarial
synthetic output/prefix/pixel/visual-grid failures. The dedicated
4-way GitHub CI repeats the same CPU-only source/fake-policy checks
on Windows Edge and Ubuntu Chromium with Python 3.11/3.12 and
pinned Playwright/Pillow. Repository-wide stdlib-only discovery
can skip the browser opt-in test. No data or gold leaves the
temporary source generation directories.

## Explicit blockers

The original source candidates remain `TRAIN_DEV_CANDIDATE_NOT_ADMITTED`
or equivalent, not official TRAIN or unseen OOD. The
`ADMITTED_TRAIN_MANIFEST_SHAS` production registry is EMPTY.
H355 independently validated legal reviewers list is EMPTY; H356
independent OOD custodians list is EMPTY. The independent rights
reviewer recruiting Issue #384 and independently custodied new blind
EN/AR OOD Issue #385 remain unresolved. Even a future *real
processor* PASS would not establish legal title, licensed rendered
screen-shot model training/weights distribution, a new approved H199
TRAIN release, source-disjoint blinded evaluation, or Qwen adapted
learned gain. Hala remains NO-GO for training/release until those
separately authenticated milestones.
