# H212 — independent bilingual original GUI research-only candidate source

**Status: code + synthetic source provenance only; NOT approved TRAIN/DEV/OOD.**
Draft stacked above H211 #232 at exact parent
`11bd9c9b0c5c92f21c52c3251165ea5c1a99cdc9`.

H209/H211 established a genuine public pre-trained 256M VLM can read
images but failed strict GUI-action JSON twice on two synthetic GUI
screens (0/2 accepted native click predictions). To move toward
eventual fine-tuning without copying previous OOD or private media,
H212 introduces a **from-scratch, source-only, fully reproducible**
96-example bilingual original graphical interface candidate generator.

## Exactly what exists

- `h212/procedural_native_gui_candidates.py` renders **96 independently
  indexed 960×720 RGB** research application screenshots entirely
  in memory using Pillow 12.3.0, with no file/media inputs.
- Four distinct self-authored *source template families*:
  **toolbar** (32), **sidebar** (32), **dialog** (16), **footer** (16).
  All show a visually distinguishable blue SAVE control and gray
  CANCEL control in varying positions. Multiple visual backgrounds
  provide context; no source index/split/gold/coordinates appear in
  the model-visible task text.
- **64** `RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED` examples (two families),
  **16** `RESEARCH_DEV_CANDIDATE_NOT_SEALED` (dialog), and **16**
  `RESEARCH_EXTRA_CANDIDATE_NOT_OOD` (footer).
  Each family is balanced 50% English and 50% Arabic instructions,
  **48 of each language** across 96 examples. SAVE and CANCEL are
  deliberately English *visual button labels* even in Arabic tasks:
  this does not establish Arabic visual-script literacy.
- Targets are computed from the newly drawn SAVE rectangle and
  converted into **strict H199/H200 native JSON one-action click
  answers**. The H199 source_task_prompt is used verbatim for
  the question, so neither response coordinates nor candidate role
  enters the model-visible task text.
- `research_multimodal_messages(index)` returns an in-memory PIL RGB
  image + task text question and the strictly parsed assistant target
  in Hugging Face-compatible chat-message structure. This is
  *preprocessing-ready*, not tokenized, trained or validated with an
  actual frozen Qwen3-VL processor. No GPU/model calls.
- `h212/RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json` pins source
  SHA and **96 distinct PNG-byte SHA-256 + 96 distinct decoded RGBA
  SHA-256** observed on the owner's Windows CPython 3.11.15 and
  Pillow 12.3.0. This source metadata includes no action point,
  private image, gold, or binary PNG. Because PNG compression and
  font rendering can vary across runtime versions/platforms,
  re-run/re-verify bytes on each target platform; SHA matches are
  *observed metadata*, not universal independent media provenance.
  No candidate's SHA has been screened against **complete** old source
  history (old RGBA and original forks remain incomplete).
- Normalized source Python SHA for this generator is frozen in
  metadata (line endings normalized for checkouts); the JSON is
  research metadata only and must never be mistaken for a
  released/source-qualified official TRAIN manifest.

## Actual owner CPU verification

The Windows owner QA venv is `D:\Hala-H212-QA` with Pillow 12.3.0
and PyYAML 6.0.3, **no PyTorch, no HF model, no media access, no GPU**.
Real checks at current H212 source:
- `tests/test_h212_procedural_native_gui_candidates.py`: **12/12**
  normal Python and **12/12** optimized `-O`: **24 executed PASS**.
- Verified 96 distinct PNG SHA and 96 distinct rendered RGBA SHA,
  source-only model-message images, English/Arabic balance,
  exact split/family roster, rendered hit-vs-opposite-button mismatch,
  deterministic repeated generation, bounded indices, malformed
  response denial and official H199 release denial.
- `python -m h212.procedural_native_gui_candidates` printed:
  `H212_NEW_SELF_AUTHORED_GUI_SFT_RESEARCH_CANDIDATES_PASS
  examples=96 train_research=64 dev_research=16 extra_research=16
  distinct_png_sha=96 distinct_decoded_rgba_sha=96
  bilingual_en_ar=true official_train_admitted=0 sealed_dev=0
  ood=0 model_steps=0`.
- The GitHub Actions Linux Python 3.11/3.12 workflow is synthetic
  Pillow-only with `HF_HUB_OFFLINE=1`; no real model,
  downloaded weights or training. **Hosted CI green not claimed**
  until it finishes on the exact commit.

## Absolute scientific qualification limits

1. The 96 new source cases are **not** officially admitted data.
   H199 `ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()` stays EMPTY,
   and `verify_real_training_release()` rejects candidate promotion.
2. DEV cases are **not sealed**, because generation code, source SHA,
   visual layout templates and label logic are public. They cannot
   serve as an official blind DEV or held-out OOD score.
3. A distinct procedural template family is *not proof of global
   old data/source disjointness*. Historical pre-H173 source/fork and
   decoded-RGBA evidence remains incomplete.
4. No official baseline improvements, new Qwen3-VL model weights,
   LoRA optimizer steps, measured model grounding on these examples,
   data licenses beyond original author-created code, native
   browser events, dev/ood evaluation, publication or release.
5. Never access consumed H173 OOD, reserved Gantt OOD, owner-private
   H188/H190/H191/H194/H197 images or sealed gold. All files in this
   PR are new source, tests, and **nonprivate research metadata**.

The next genuine scientific step is an **independently source-qualified
training release** and a **free, documented processor/model parity
check** before attempting a LoRA on Qwen3-VL; this H212 source alone
does not waive those gates.
