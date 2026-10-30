# H234 — native bilingual text entry contract (CPU-only, synthetic)

**Research engineering only. No admitted TRAIN/DEV/OOD, no model call,
no optimizer step, no checkpoint or release.** This addresses the concrete
H232 original-app reachability gap: H199 model action vocabulary only
covered click, drag and key_press, while real original GUI workflows need
visible-field text entry in English and Arabic.

## Safe compatibility extension

- `h234.native_type_text.parse_actions_v4` preserves the exact existing
  H199 actions and abstention semantics via delegation. New
  `{"kind":"type_text","point":[x,y],"text":"..."}` is one screenshot-
  grounded action, never a DOM id or CSS selector, and cannot be bundled
  with other actions or arbitrary shell/keyboard macros.
- Point is the existing 960 × 720 screenshot pixel coordinate system.
- Text must be 1–160 Unicode code points, at most 360 UTF-8 bytes,
  already NFC-normalized and trimmed; no C0/C1 controls, format codes,
  surrogates, private-use code points, carriage returns or newlines.
  This retains ordinary Arabic and English input while rejecting
  unrenderable/bidi-control payloads.
- Duplicate JSON keys, NaN/Infinity, extra fields, oversized JSON and
  unexpected action kinds fail closed.
- `source_task_prompt_v4` extends the original H199 screenshot-only
  prompt with the new visible-field entry action. The original H199
  contract, allowlist and other studies are NOT edited.
- `synthetic_sft_messages_v4` uses the H199 multimodal structure only
  with fictional task/action pairs. It does NOT import any images or
  admit optimizer gradients.
- `replay_local_type_text` accepts only an already-open `file://`
  Playwright page. It clicks a screenshot pixel and uses genuine
  Playwright keyboard text insertion, not hidden DOM or a selector in
  the model output. The caller is responsible for browser isolation
  and verifying the UI outcome. No automatic dataset admission.

## Owner Windows reproduction

From the repository root:

```powershell
python -m pytest -q tests/test_h234_native_type_text.py tests/test_h199_native_action_sft.py
python -O -m pytest -q tests/test_h234_native_type_text.py
```

The synthetic QA uses an ephemeral self-authored `file://` input fixture,
plus optional already-installed Microsoft Edge or Chromium. It enters
English and Arabic via real browser mouse/keyboard actions with network
requests blocked. Hosted Python-only CI may skip the optional native
browser test if no browser is preinstalled. No downloaded browser, donor
assets, model, GPU or private owner image is needed.

## Explicit scientific boundaries

The contract cannot train a model. H199
`ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()` remains unchanged; H212 research
candidates remain excluded from optimizer gradients by H222. H173 original
OOD has been permanently consumed and must never be reopened.

The next milestones are independent actual original source provenance
and history, source-disjoint admitted TRAIN/DEV, fully legal UI assets and
sealed new OOD, a separately qualified training-release manifest, then a
free 2B LoRA training smoke and new independent evaluation. H232's original
Awesomplete QA establishes native app reachability only, not native model
performance or admission of real examples.

No main-branch changes, PR merge or model release is part of H234.
