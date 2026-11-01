# H342 — Original SignalDesk visual-neighborhood audit

**Status:** source-only exploratory visual diversity diagnostic, not Qwen evidence.

H341 established that 144 public synthetic scenes from RailKit and SignalDesk
remain just **two** source families. H342 tests a different empirical question:
how often are the 256 *RGB-distinct* SignalDesk candidate frames similar under
a fixed, explicitly lossy global visual fingerprint?

The source of all pixels is pinned original H338 SignalDesk HTML Git blob
3d787aa3aafb999d342cbd3685c20d0f34d5fcd8. The existing H338
offline Playwright harness first validates **all 128 EN/AR native two-step
source tasks**, 256 decoded-RGB-unique states, strict screenshots and focus.
Only then does one transient trusted in-memory sink compute a 32x32
grayscale adjacent-column difference hash (1024 bits). The result is a
fixed-band histogram across 32,640 within-family frame pairs, plus coarse
same-language versus cross-language near-pair counts. No PNG, task,
native action, coordinates, per-scene score, or individual fingerprint is
returned, logged or written. No model is loaded or invoked.

**Important limitations:** perceptual distance is not semantic equivalence;
zero hash distance can occur for different RGB images, and large hash distance
does not establish source independence. Both languages, both native action
stages, and every visual variant still belong to **one** SignalDesk source
family. This does not establish cross-family DEV/OOD independence, legal title,
rendered screenshot ML training rights, H199 TRAIN admission, or BASE/ADAPTED
model gain. It must not be used to justify a random scene-level TRAIN/DEV split.

To run the complete public source-only browser audit, use the pinned free
GitHub Actions H342 workflow on Ubuntu Chromium or Windows Edge. The default
CPU test suite uses invented-only data, except a tiny two-image Pillow
decoder check; the browser test is opt-in. No GPU, Kaggle, private media,
weights, optimizer, deployment, or external network source is involved.

**Scientific next gate:** obtain independent written source and rendered-image
ML training rights, then separate family-disjoint lawful TRAIN/DEV and new
never-exposed blind OOD custody before prospective genuine paired
BASE/ADAPTED Qwen native task-state inference with abstention safeguards.
H173 remains sealed; H293/H294 safety barriers remain binding.
