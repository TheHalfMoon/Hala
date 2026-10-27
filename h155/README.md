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

## Verified original action supervision — SOURCE ONLY

Owner Windows captured **48 new TRAIN and 12 new DEV** original MiniWoB++
960×720 source screenshots with supervised click/type/select/submit JSON.
All 60 reference actions completed with original application reward,
all 60 image SHA values are distinct and disjoint from the **21 earlier
H154 source images**. The independent owner audit passed and froze
manifest SHA-256
`ad3fc8c74d0601b55f941ce77c8ee57b34101d7f5dfeaf8cc28904161e65ccd1`.
Private artifacts are `private-h155/source-train-dev-v1/`, not on GitHub.
This is **single-app, English-only TRAIN/DEV supervision**, not model
training, unseen-app evaluation or a model performance result.

## Remaining real work

1. **Additional TRAIN:** expand source-only GUI supervision beyond the
   current MiniWoB++ originals to at least one other licensed application,
   and add bilingual tasks. Keep SHA-disjoint TRAIN/DEV/eval splits.
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
