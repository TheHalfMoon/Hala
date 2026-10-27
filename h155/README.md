# H155 — strict image-to-action protocol for cross-app computer use

**Engineering foundation only.** No new model has been trained, run, or
shown to succeed on MiniWoB++ here.

H153 established that the existing external frozen Qwen3.5-2B LoRA could
predict START/visible Kanban column-heading points on one original React
app, reaching 15/16 persisted drags with the improved H152 native action
controller. That model is **NOT** authorized to be represented as capable
of clicking buttons, filling forms, selecting dropdowns or entering text.

H154 separately qualified the MiniWoB++ original application, with
12/12 **reference-only** successful browser actions from three task
families. No model predictions produced that score.

## Admitted new model output syntax

Only after a **new trained H155 model checkpoint, explicitly identified
and SHA-pinned**, model text must be a plain JSON object with one
`actions` array of 1–5 typed actions:

- `{"kind":"click","point":[x,y]}`
- `{"kind":"type","text":"value"}`
- `{"kind":"select","text":"visible option"}`
- `{"kind":"submit","point":[x,y]}`
- `{"kind":"abstain"}` (standalone)

Valid coordinates must be finite numbers inside the **960×720** image.
Values are bounded to 128 non-control characters. No extra fields,
arbitrary JavaScript, DOM selectors, screen-reading APIs, URLs,
inferred oracle fields or code execution are accepted.

Task-specific sequence checks:

| Task | Allowed native action sequence |
|---|---|
| `click-test` | click |
| `enter-text` | click, type, submit |
| `choose-list` | select, submit |

A parsing failure is a **failed/unsupported prediction**, never
repaired with original-app oracle coordinates or text. Frozen H142
START/visible heading roles are treated as
`UNSUPPORTED_ACTION_ROLE_ABSTAIN` on all H155 tasks.

## Remaining real work

1. **H155 TRAIN:** create newly generated source-only original GUI
   training labels and screenshots across action families and at least
   two independent applications, with true MIT/source pin evidence.
   Keep split IDs and screenshot SHA values disjoint from H137–H154.
2. **H155 DEV:** independently freeze balanced validation tasks.
   Determine if the model can genuinely emit strictly typed actions,
   not just source/reference UI geometry.
3. **H155 BLIND:** only after model and parser are frozen, create new
   owner-private one-time evaluation tasks on unseen screenshots and
   native Chromium. Evaluate real task success from final state/reward,
   not parser acceptance or source oracle. Include abstentions in
   denominator.
4. **Compute:** only free, permitted non-commercial Kaggle/Colab;
   no paid fallback. Preserve provenance, SHA receipts and negative
   results. Do not unseal H140 heldout OOD blind.

Local H155 contract tests: **5/5 PASS** and combined focused
H150–H155 tests: **20/20 PASS**. They qualify the parser only.

No Hala flagship 35B training, model promotion, deployment,
repository main merge or release.
