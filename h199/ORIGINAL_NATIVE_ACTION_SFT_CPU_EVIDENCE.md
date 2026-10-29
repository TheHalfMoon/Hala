# H199 — Native GUI action SFT contract, synthetic CPU smoke only

**Engineering milestone:** copy the existing H161 visual native
click/drag SFT JSON/prompt/completion-loss pattern and extend it with
a screenshot-grounded `key_press` action for the genuine original
interact.js slider tasks introduced by H193/H194. This closes an
**action vocabulary and training-preparation compatibility gap**,
NOT the data qualification or free GPU training blocker.

## Supported single-step model action vocabulary

The only valid predicted shape is one exact JSON document with a
one-element `actions` array. All coordinates are 960×720 screenshot
pixels, never hidden DOM ids, selector, canvas metadata or owner gold.

| Kind | Valid exact action | Allowed use |
| --- | --- | --- |
| `click` | `{"kind":"click","point":[x,y]}` | Existing native Pickr/Datepicker and H161 click |
| `drag` | `{"kind":"drag","start":[x1,y1],"end":[x2,y2]}` | Existing native interact.js drag and H161 drag |
| `key_press` | `{"kind":"key_press","point":[x,y],"key":"ArrowRight"}` | Visually focus a slider before native keyboard input |
| `abstain` | `{"kind":"abstain"}` | Model uncertainty or unsupported action only, NEVER training gold |

Keyboard key allowlist: `ArrowLeft`, `ArrowRight`, `ArrowUp`,
`ArrowDown`, `Enter`, `Escape`, `Tab`, and `Space`.
No free-form text, arbitrary shortcuts, shell commands or compound
keyboard macros. Non-finite, boolean, out-of-screen or negative
coordinates, zero-distance drags, extra JSON fields, multiple actions
and oversized model responses fail closed.

## Prompt and multimodal loss

`source_task_prompt` specifies only the model-visible original
960×720 screenshot and human task. It instructs the model to avoid
DOM/hidden HTML and abstain when necessary. The **synthetic-only**
message composer creates an image-bearing user message followed by
a typed native assistant action.

`assistant_completion_mask` takes two actual token-ID sequences:
the model/processor-produced image+task user prefix, and the full
prefix+assistant action. It requires exact token-prefix equality
and a nontrivial assistant completion, labels EVERY user-prefix
token `-100`, and supervises ONLY assistant completion tokens.
This is the portable **H161 implementation contract**. Actual
Hugging Face processor parity and GPU training loss were NOT tested
or claimed; that requires a source-qualified future training release
and model run.

## Actual Windows results, October 29, 2026

Owner device: Microsoft Windows, CPython 3.11.15. Source SHA
`eda241a42da3511f0e1466d3f75cc86b069327d6`:
**14/14** typed action / prompt / token masking / release refusal
unit tests PASS normally, **14/14** with `python -O` PASS;
`python -m h199.native_action_sft` prints:

    H199_SYNTHETIC_CPU_NATIVE_SFT_CONTRACT_PASS
    synthetic=3 CLICK_DRAG_KEYPRESS ABSTAIN_FALLBACK_ONLY
    NO_REAL_TRAIN_NO_MEDIA_NO_OOD_NO_GPU

The synthetic-only examples cover click, drag, keyboard input
and assistant-only supervision; all are hardcoded fictional action
coordinates, **not** owner-private source gold. No PNG, gold, old
scored OOD, Gantt tentative OOD, adapter, Kaggle or Colab resource
is opened.

## Absolute independent training/data admission gate

The public H199 training admission allowlist
`ADMITTED_TRAIN_MANIFEST_SHAS` is **intentionally empty**.
`verify_real_training_release` rejects every real dataset release,
including a self-declared `official_admitted=true` or bogus
`complete_history=true`, until a newly verified frozen manifest
SHA has been independently preregistered, reviewed and pinned in
a future source-and-history-qualified code version.

All H188/H190/H191/H194/H197 20 real original GUI images remain
`research-only, TRAIN/DEV/OOD_eligible=false`.
Their old source/fork/npm and decoded-RGBA historical coverage
remain incomplete, H173 scored OOD is permanently consumed and
Gantt tentative OOD remains unopened. No model weights, loss values
or accuracy improvements were produced by H199.

**Next genuine model milestone:** add more new original application
and task diversity, establish truly admissible TRAIN/DEV splits
with sealed DEV gold, then take this contract into a free
2B LoRA/HF processor **real** training smoke and external
independent OOD once. Never claim source-only QA equals training.
Keep all engineering Draft, never merge main/publish/paid GPU.
