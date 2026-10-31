# H261 — four-family historical source/media overlap research

**Owner-only real Git analysis, public Hala-authored code and invented tests.**
No original donor bytes, source receipts, private raw hashes, images, H173
scored OOD, model weights or gold are published. No dataset admission.

H260 added two real original offline Pomodoro candidates. H261 examines
**all locally reachable Git refs** in four separately pinned original Git
repositories: H250 To-Do, H251 LifeOS, H260 Pomodoro A and Pomodoro B.
It checks pairwise historical Git blob byte identity for source and media,
and 7-token source-shingle Jaccard similarity (90% threshold) for
near-identical source files. It never displays original source or media;
its output contains counts and similarity percentages only.

The original upstream MIT license does not prove rights to every asset.
A zero overlap among these four sources does not prove independence from
other past Git forks, npm mirrors, deleted objects, older Hala data or
visually identical images with different encoding. Git blob equality does
not replace decoded-RGBA exclusion checks. A high token score can be
boilerplate rather than common authorship. No automatic acceptance occurs.

## Private Windows reproduction

Use the clean, pinned original source checkouts **outside the Hala repo**.
Pass four exact local paths and privately verified 40-character HEAD pins
to `python -m h261.historical_family_collision`:

```powershell
python -m h261.historical_family_collision `
  --h250-repo PRIVATE_H250_CLONE --h250-pin PRIVATE_H250_PIN `
  --h251-repo PRIVATE_H251_CLONE --h251-pin PRIVATE_H251_PIN `
  --h260a-repo PRIVATE_H260A_CLONE --h260a-pin PRIVATE_H260A_PIN `
  --h260b-repo PRIVATE_H260B_CLONE --h260b-pin PRIVATE_H260B_PIN
```

All source/media Git blobs are read only on the owner Windows machine and
kept in process memory. The script refuses changed HEAD, dirty working tree,
unexpected upstream, malformed history, oversized source or unbounded
inventory. Only bounded aggregate counts are printed; no receipts are
written. Hosted CI executes invented fixtures only.

**Scientific boundary:** H199 admitted TRAIN allowlist remains EMPTY; H173
32-case scored OOD remains SEALED; H212 procedural scored data is barred.
No Qwen3-VL inference, LoRA optimizer step, model accuracy or task-state
success was measured by H261. Next is truly independent complete historical
and transitive rights qualification, new sealed blind OOD and authorized
free noncommercial GPU baseline vs LoRA comparison.

## Actual owner-Windows result (2026-10-31)

All four clean original checkouts passed exact private HEAD, remote and status
preflight. Locally reachable commit counts were **1, 1, 9, 2** (H250, H251,
H260A, H260B). Unique historical source Git blob counts were **3, 23, 3, 5**;
media blob counts were **0, 10, 1, 0**, respectively.

Across the **six pairwise family comparisons**, there were **zero identical
source blobs**, **zero identical media blobs**, and **zero historical source
file pairs with token-shingle Jaccard >=90%**. The maximum measured pairwise
source-file token Jaccard was **8.13%**. This is meaningful local evidence
against direct byte-identical source/media reuse among these four known
checkouts, not a proof of globally independent authorship or visual pixels.

H251's ten historical media blobs still require independent rights review;
H260A's historical screenshot and five current inline SVG elements remain
unqualified. Historical deleted Git objects, all forks/npm mirrors, all
prior Hala families, full PNG/RGBA exclusions and new sealed blind OOD remain
unproven. **All four candidate families remain quarantined.**
