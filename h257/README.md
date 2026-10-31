# H257 — private historical visual exclusion integration

**DENIAL ONLY. No TRAIN, DEV, OOD admission, model inference or optimization.**

H256 recovered a surviving pre-H173 PNG/RGBA exclusion receipt on the owner
Windows machine. H257 joins those private exclusions to **actual on-disk**
H250 and H251 original-app research PNG frames, validating both receipt hashes
against the current PNG bytes and H231-compatible decoded RGBA pixel hashes.
The output is aggregate counts only. No private image, SHA, path, prompt,
source pin, gold, or reference is committed or printed.

Owner Windows result (2026-10-31): 24 original research frames were rehashed
from private PNG files against 144 surviving historical PNG exclusions and 137
historical decoded-RGBA exclusions. **Zero known historical overlaps** in
both modes, but only **23 distinct candidate PNG and RGBA hashes**: one
repeated candidate frame was discovered. This is not 24 independent visual
examples. The private aggregate receipt remains outside Git.

H250 and H251 are different original MIT app source families but still
**quarantined**: full historical source/fork/npm/deleted-media provenance is
unproven, and H251's loaded logo has unresolved transitive media rights.
Even a zero-collision result proves only no collision against *surviving*
known H256 exclusions. It does not approve a split or a model release.

The CLI fails closed for missing, malformed or altered screenshots,
counts repeated screenshots as non-independent, and rejects missing source
receipts, a changed H256 inventory, sealed H173 path references,
and outputs in Git or pre-existing destinations. It does not traverse H173 or
reopen any scored OOD. Historical receipt fields remain owner-private.

Run on authorized Windows with a **new** owner-private output folder outside
all Git worktrees, never overwrite previous receipts:

```powershell
python -m h257.private_visual_collision_gate `
  --historical-receipt $env:HALA_H256_PRIVATE_RECEIPT `
  --h250-receipt $env:HALA_H250_PRIVATE_RECEIPT `
  --h251-receipt $env:HALA_H251_PRIVATE_RECEIPT `
  --private-output $env:HALA_H257_NEW_PRIVATE_OUTPUT
```

Tests use invented in-memory PNGs and synthetic manifests, never owner data.
Do not equate synthetic CI with a real model, training eligibility, full
visual independence or rights clearance. H199 admitted TRAIN remains empty;
H173 32 scored OOD cases are permanently sealed.
