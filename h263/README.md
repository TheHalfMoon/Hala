# H263 — One-step native browser action executor (research only)

This module connects the strict H199 one-action model-output JSON contract to **actual browser mouse/keyboard events** at a fixed 960×720 viewport. It supports click, drag, key_press, type_text (only into a pre-focused editable field), bounded scroll, and abstain. The executor returns only the action kind and whether a browser call completed. It never reads screenshots, original gold, reference receipts, or task labels and never reports task success.

`new_offline_context(browser)` constructs an offline, fixed-viewport Playwright browser context with downloads and service workers disabled. The caller must load an authorized local application, ensure no private screenshots are transmitted or published, and keep task-state oracles separate from model-visible inputs. No GPU, weights, remote service or dependency download is used by the public tests.

## Evidence and limitations

- **Owner Windows real Microsoft Edge**: 10/10 aggregate browser state/safety checks on the H262 first-party Stockroom, including real click mutation, drag/drop mutation, Arabic typing into a separately focused field, Escape, scroll, abstain, and no external network/page errors. The private browser runner saved no screenshots, task references or gold and is **not committed**.
- **Public invented contract tests**: 14/14 pass; cover exact H199 parse, duplicate-key rejection, viewport guard, five action types, focus safety, abstain, and no raw-action data in the result.
- This is **not model inference**: the browser smoke used authored actions, not Qwen3-VL predictions. It is **not an independent third-party OOD**; H262 Stockroom is first-party, not admitted to TRAIN/DEV/OOD.
- A completed browser call does not mean the user's task succeeded. Future blind evaluation must score actual final GUI state with a separate private oracle and independent untouched tasks. Do not feed DOM, gold, state or source to the model.
- H173's consumed 32-case scored OOD remains permanently SEALED; H212 procedural scored data remain barred. H199 official TRAIN manifest remains empty. H15 free Colab T4 real two-step LoRA optimizer evidence is valid engineering evidence, not task accuracy.
- Public PR contains only SOURCE/DOC/INVENTED SYNTHETIC TEST and CI, never actual screenshots, private labels, raw predictions, private SHA, donor bytes or weights.

Run source-only tests: `python -m pytest -q tests/test_h263_native_browser_executor.py`.
