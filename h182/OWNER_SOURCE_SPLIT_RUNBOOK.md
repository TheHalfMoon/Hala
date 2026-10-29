# H182 — preregistered source-disjoint TRAIN/DEV research roles

**Status: PROSPECTIVE SOURCES ONLY. NO datasets, model training, or OOD scoring.**

## Source roles (not training split admission)

| Immutable public original app | Proposed research role | State |
| --- | --- | --- |
| Pickr (`simonwep/pickr`) | TRAIN prospect | Original source-only H179 and two-scene H180 native QA; no dataset admission |
| interact.js dropzones (`taye/interact.js`) | DEV prospect | Original native drag/drop H175 QA; no dataset admission |
| medium-zoom (`francoischalifour/medium-zoom`) | Research-exposed, unassigned | Source already inspected, not a sealed OOD |
| Frappe Gantt (`frappe/gantt`) | Tentatively reserved OOD source identity | **No GUI build, browse, source task images or model evaluation permitted** |

The above is an immutable **source identity plan**: `h182/source_role_plan.py`
copies exact H174/H179 and H181 Git/HTML/MIT license source pin records.
Its role names intentionally have `PROSPECTIVE` or `NO_IMAGES`
qualifiers; they are **never** formal `TRAIN` / `DEV` / qualified OOD
dataset partitions. Gantt is reserved **as a source ID only**, not proof
of complete OOD novelty; it may later be rejected if a previous hidden
fork, source archive, npm bundle, deleted checkout or other family alias
is discovered.

Both sources in the proposed train/dev roles were exposed through earlier
native proof research; they must not be called truly untouched OOD.
Source-disjointness requires different original upstream repos and Git
source identities (not simply independent screenshots of the same repo).

## Owner Windows reproducible source-only command

Run from this exact H182 branch in the authorized local Windows owner
checkout; change paths only to point to already existing pristine donor
source repos:

    python -m h182.audit_owner_split_plan --historical-root C:/Users/Shehr/work/Hala --pickr-donor D:/Hala-H174-Donor-Pickr-SourceOnly --interact-donor D:/Hala-H174-Donor-Interact-SourceOnly --mediumzoom-donor D:/Hala-H174-Donor-MediumZoom-SourceOnly

The command composes **existing H179 and H181** source and owner-history
checks, and requires all three prior original donor source byte pins
together. It checks 22 surviving prior original Git checkouts, the
previous known Git/npm identities and independently pinned original
HTML and license files. It verifies the *same fingerprint of every
observed historical repo/Git SHA pair* in both donor and reserved OOD
screens; equal counts alone cannot catch a source revision changing
between scans. No owner private source names, paths, labels, task
payloads or screenshots are printed.

A local-only no-donor mode is available for synthetic validation. It
prints `original_donor_byte_pins=0` and is not a provenance qualification.
Either mode always returns **no training/evaluation approval**.

## Required gates before any data capture or training

1. Independently assemble a **complete historical original source family
   registry** from prior provenance, including deleted sources, copied
   original code, npm archives and forks. The 22 surviving owner Git
   checkouts are a *lower bound*, not complete proof.
2. Independently audit prior owner-only screenshot PNG/decoded RGBA SHA
   registries from existing manifests **without reopening consumed
   evaluation images or gold**. Missing or untrusted history blocks
   data admission.
3. Capture genuinely new independent original-app TRAIN and DEV scenes,
   with real native outcome verification, privacy/source boundaries,
   balanced bilingual tasks and independent exact decoded RGBA replay.
   **No Gantt captures or other OOD access** during training/DEV.
4. Gate each new case with the H174 source/visual SHA firewall and
   authenticated owner-side history evidence, then allow a new training
   run on zero-cost free GPU only if all independent checks pass.
5. Freeze code, model, seed, hyperparameters and evaluation protocol;
   only then assess independently qualified completely untouched OOD
   once and report paired uncertainty. H173 scored v3 is permanently
   consumed, not to be re-run, used for training or used for selection.

**Do not merge into protected main, do not publish source/media/model
weights, do not incur paid costs, do not require human review.**
