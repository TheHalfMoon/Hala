# H178 — offline historical original-source Git checkout inventory

**Research-only identity census. Not a model checkpoint, evaluation, or data admission.**

## Why H178 exists

H177 independently pins nine previous source repositories and 22 known old
application IDs. On the authorized Windows owner machine, a **read-only
metadata-only** inventory revealed further local historical Git source
checkouts outside that public minimum. Some belonged to older private
research and original-app studies. Their previous presence must not silently
become a "brand-new untouched" OOD source.

This module works exclusively on Git repository identity, not browser media.

## Strict traversal boundary

Given one or more explicit local roots, the scanner considers only:

- direct phase directories named private-hNN / private-hNNN or
  Hala-HNN-Private / Hala-HNNN-Private
- immediate children named upstream, vendor-apps, vendor, sources, donors
- immediate Git checkout directories with a real .git **directory**

It rejects symlinks, Windows reparse points/junctions and file-backed .git
worktree pointers. It does not descend into captures, images, old OOD,
results, JSON manifests, training data, node_modules, caches or model
directories. Git commands read only local config/rev-parse metadata and
have strict timeouts. No network fetch, install, model inference or media
read/write is performed. It handles up to 100 phase folders and 400 source
checkout directories per invocation; larger histories fail closed.

GitHub remote identities must use a canonical
https://github.com/OWNER/REPOSITORY or git@github.com:OWNER/REPOSITORY
form with no embedded credentials. Original Git HEAD is checked and its
repository top-level must match the scanned checkout directory.

## Owner-only execution (do NOT publish results containing source IDs)

Inventory-only mode, with local path as an example:

    python -m h178.scan_owner_sources --historical-root C:/PRIVATE/Hala

Multiple --historical-root options are permitted for separate owner-only
volumes. The program prints only checkout count, distinct repository count
and a SHA-256 fingerprint of source identity metadata. No source name,
remote URL, original path, labels or screenshots are printed. This mode
**cannot qualify a new dataset or model**, even when it exits 0.

To cross-check a real owner historical registry as well, use all four
owner-only metadata manifests from H176/H177:

    python -m h178.scan_owner_sources --historical-root C:/PRIVATE/Hala --candidates C:/PRIVATE/candidates.json --prior-pixel-hashes C:/PRIVATE/old-hashes.json --prior-source-history C:/PRIVATE/old-source-history.json --paired-source-history C:/PRIVATE/paired-history.json

All four files must be present; partial submission fails closed. This
check composes the H174, H176 and H177 source-history gates and rejects
both a new candidate's reuse of ANY observed historical checkout
repository and an old repository not represented in the owner historical
registry. Only the H137 npm exception already anchored in H176 is allowed
outside the legacy Git-repository list.

## Limitations and next requirements

Present local Git checkouts cannot prove **complete** historical coverage:
some used sources may have been deleted, moved, stored as npm bundles,
downloaded archives, copied files, or may have been used on another device.
A source may also have been inspected without a local Git checkout.
Even full H178 PASS therefore *always* reports historical completeness
unproven and training/evaluation not approved.

Before new Hala training, independently verify the full past source
registry from source-only historical evidence, exact licenses and frozen
original source checksums; construct fresh family-disjoint TRAIN/DEV
examples with native behavior and SHA-verified independent pixel replay;
reserve unseen OOD before training and never reuse consumed H173 v3 or
earlier DEV/OOD. Only then proceed to permitted free-GPU adaptation,
frozen candidate and genuinely untouched one-shot OOD assessment.

No publication, paid compute, protected-main merge, model release or
human-review prerequisite is part of this Draft.
