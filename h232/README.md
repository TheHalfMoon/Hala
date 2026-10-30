# H232 — pinned original Awesomplete native GUI QA (research only)

**Draft only (public-source code, no owner-private evidence). Never TRAIN/DEV/OOD admission, model inference, optimizer, release or merge.**

## Genuine original upstream source

- Original MIT source: canonical GitHub `nudeui/awesomplete` (immutable repository ID `30987756`), Git commit `596bc5d5916266437970ecbf65434077d3950a3d` (`gh-pages`). The historic `LeaVerou/awesomplete` URL redirects to this SAME repository ID; both aliases are checked against the known historical-family floor. A cloned Git remote may still show the old URL. GitHub API commit and repository identity were independently verified 2026-10-30.
- `LICENSE` SHA-256: `3358ef750899fd62ec44637f2a23719a89bb7a8a0b4225d72488d2483359cf1d`.
- Original upstream `test/fixtures/options.html` SHA-256: `907c1a85936ac6f0a0eaaa7ad4867d44fbd7c3d2dcc179dfccd405f87b098c07`.
- Original `awesomplete.js` SHA-256: `5f5b922b7945bf4a58487080ef075bf7e2734c953063886276cd386d760803f8`.
- The exact original fixture is a fragment with an input and data-list, not the production demo. The original `index.html` contains third-party analytics/social embeds, so this QA deliberately **does not load** it. The harness injects the pinned original JS and calls its public constructor on the unmodified original test fixture. This is **real original library behavior under QA instrumentation**, not an unchanged original demo, new labeled dataset, or Hala model prediction.
- No npm install, bundler, remote resource, pretrained weights, or external asset is required.

## Owner Windows reproduction

Use only the authorized Abdulaziz Windows device and an isolated, clean, pinned clone. Do not touch old H173 media/gold or any previous owner cohorts.

```powershell
cd D:\Hala-H232-Worktree
python -m pytest tests/test_h232_original_awesomplete_native.py -q
python -m h232.original_awesomplete_native --source-root D:\Hala-H232-Awesomplete-Original-20261030 --output D:\Hala-H232-Private\awesomplete-native-01
```

The output directory must not already exist, must be outside every Git checkout and is private owner-local only. The script checks clean exact Git commit, origin, root MIT license, exact original HTML and JS hashes, known H173/H176/H137 source-family exclusion and scoped `file://` routing. Every unexpected network/out-of-root request fails closed with `INCOMPLETE_NO_ADMISSION`.

The native QA runs two independent Edge Chromium contexts: English typing `Da` must open the original autocomplete (`aria-expanded=true`), choosing original suggestion `Data` must set the real input value and close it; Arabic `مرحبا` must be accepted as text without inventing an Arabic suggestion. It records before/after PNG and decoded width/height-bound RGBA SHA-256, but **never** commits or uploads those private screenshots. Arabic entry is *not* bilingual suggestion accuracy.

## Observed owner Windows evidence, 2026-10-30

At pinned source SHA above, original source checkout remained clean. The actual Microsoft Edge replay succeeded in **2/2 independent contexts** with **zero network attempts**. Image and decoded-RGBA SHA-256 receipts are retained locally on the owner device, and omitted from this public source-only PR:

- Before screenshot: owner-local receipt only (not checked in)
- English selected: owner-local receipt only (not checked in)
- Arabic text entry: owner-local receipt only (not checked in)

Private output: `D:\Hala-H232-Private\awesomplete-native-01`, 6 local PNG files plus receipt JSON. Only source-hash and outcome receipts belong in this private PR; no PNG/gold/owner metadata uploaded.

## Limits and scientific firewall

This independently pins **one new public source** and proves an authored QA path through genuine original JS in Windows Edge. It **does not prove** complete historical source/fork/npm inventory, decoded historical RGBA inventory, absence of this family from all prior private experiments, complete transitive license/media rights, or original-app full demo behavior. It is not an admitted TRAIN/DEV/OOD example and never updates H199. H212 gradients remain blocked by H222; H173 owner OOD remains permanently consumed (2/32 vs 3/32; exact McNemar p=1.0, not significant).

GitHub Actions on the private repository remain blocked at account billing/spending infrastructure before job steps (H231 run 41075493441). Do not rerun blindly, pay, make public or claim hosted CI green. Qualification here is **owner-local Windows only** until zero-cost hosted CI actually executes.
