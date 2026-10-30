# H254 — Actual offline browser media use and Git provenance gap

**Scope:** public source code, synthetic tests and this documentation only. No original donor assets, screenshots, private source receipts, user data, historical OOD, model predictions, or weights enter Git.

H253 discovered referenced media in intact original MIT apps. H254 adds a materially different measurement: launch the actual pinned original HTML in an ephemeral Microsoft Edge context with H243 browser-level offline protection **before** creating a page, block external requests and out-of-root file requests, and inspect only aggregate DOM image loading. No screenshot is taken. This is a browser observation, **not** an authorship/license determination, an OS-level network sandbox, or model evaluation.

## Reproducible Windows commands

```powershell
python -m h254.original_media_runtime_evidence --repo D:\Hala-H251-Original-LifeOS --sha b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c --entry index.html
python -m h254.original_media_runtime_evidence --repo D:\Hala-H250-Original-simple-to-do-list --sha 009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff --entry index.html
python -m pytest -q tests/test_h254_original_media_runtime_evidence.py --basetemp D:\Hala-H254-Pytest-Fresh
```

## Real original-app results (owner Windows, 2026-10-31)

| Observation | H251 LifeOS Arabic | H250 Simple To-Do |
|---|---:|---:|
| Actual DOM `<img>` elements | 1 | 0 |
| Actual successfully loaded local DOM images | **1** | 0 |
| External HTTP requests blocked by route | 2 | 1 |
| External HTTP responses observed | 0 | 0 |
| Out-of-root file requests / WebSocket events observed | 0 / 0 | 0 / 0 |
| H253 static local media references | 2 | 0 |
| H253 inline SVG elements | 0 | 2 |
| Dataset admission | **NO** | **NO** |

**Material conclusion:** the H251 local logo was not merely referenced; one original public local image loaded successfully into the actual Arabic **and independently checked English** app DOMs in offline Edge (1/1 each). This creates a concrete visual-rights dependency for H251's research captures. The favicon is statically referenced but is not proven loaded by this `document.images` check. H250's inline SVG paths also remain without separate authorship proof.

The H251 original shallow checkout cannot prove complete asset history. Separately, public GitHub commit-path history for `3mmar404/LIfe-Os-Free` identifies both `assests/logo3.png` and `assests/icon.ico` in the repository's initial zero-parent `Add files via upload` commit `0e9cbb6190ca7d4817587d52acc16ec47782d17d`. A root MIT license and initial bulk upload are **not independent evidence of chain of title** for those image bytes. This is an unresolved provenance gap, **not** an allegation of infringement.

Browser offline emulation plus routing is defense-in-depth, not a proof against all native/UDP/WebRTC traffic. This check does not inspect CSS-computed icons, canvas, dynamic images after interactions, or full browser cache. The public original Git checkout is read by the browser only; no media bytes are committed or printed. All counts are research-only. No H173 sealed scored OOD, old owner media, or private historical SHA was read.

**Admission gates remain closed:** complete old original Git/fork/npm source and decoded RGBA histories, independent rights for all visible assets, clean multi-family TRAIN/DEV and fresh untouched OOD. H199 admitted TRAIN allowlist remains empty, H212 procedural optimizer ban remains, and prior H15 LoRA engineering is not a qualified Hala GUI model. The next step is obtaining defensible media rights or excluding visually dependent candidates, not claiming training readiness.
