# H249 — bounded text and scroll for the screenshot-only Hala action contract

**Status: public source / invented synthetic CPU tests only; NOT a real model run, native
host execution, TRAIN/DEV/OOD admission, LoRA, or benchmark improvement.**

H199's original one-step action contract could click, drag, or press a small
set of keys. That is insufficient to enter Arabic or English task names or to
scroll real documents/boards. H249 closes the **proposal schema** gap:

- `{"actions":[{"kind":"type_text","text":"أضف مهمة جديدة"}]}`
  proposes one bounded Unicode text insertion **only when an input is
  already visibly focused from a previous step**; it cannot also click/focus.
- `{"actions":[{"kind":"scroll","point":[480,350],"dy":240}]}`
  proposes one vertical wheel scroll over a visible screenshot coordinate.
  A positive `dy` moves downward and a negative `dy` upward; the signed
nonzero integer vertical distance is at most 720 pixels.
- Existing click/drag/arrow-key/abstain actions continue to parse.
  Strict one-element action arrays and H199 release firewall remain intact.

The proposed text is limited to 160 Python characters and 512 UTF-8 bytes;
all Unicode control/format, bidi override/isolate, surrogate, private-use and
unassigned characters are refused. Arabic letters and combining marks remain
valid. There is no selector, HTML, DOM, script, clipboard, arbitrary shortcut,
multi-action macro, or shell execution pathway. The code is **parser and prompt
only**; it never presses an actual key, submits a field, or confers authority.

## Downstream engineering still necessary

A future trusted host must re-observe focus and screen context, decide whether
typing or scrolling is authorized, execute a **single** action, recapture the
result, and audit whether the requested state actually changed. Do not treat
this source-only action parser as a secure executor or independently verified
Qwen3-VL output. Source/model tests and training remain distinct.

The H199 `ADMITTED_TRAIN_MANIFEST_SHAS` allowlist is intentionally empty. Prior
H173 scored OOD remains sealed; research H242-H248 data are not admitted,
and this PR does not load media, pretrained weights, external datasets, or GPU.

## Reproducible CPU checks

```powershell
python -m pytest -q tests/test_h199_native_action_sft.py tests/test_h249_bilingual_text_scroll.py
python -O -m pytest -q tests/test_h249_bilingual_text_scroll.py
python -m compileall -q h199 tests/test_h249_bilingual_text_scroll.py
```
