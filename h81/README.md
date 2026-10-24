# H81 — Verified reuse of the original H71 TRAIN stage after interruption

**Parent:** H80 Draft PR #64 at SHA `8d874f4324afa8d04c312502f9d894502d77ef5f`.
**Stage:** CPU-only code and synthetic tests. No T4 training, private provider upload, optimizer update or adapter-quality claim.

## Source-first recovery problem

H79 correctly moved full H67 SHA-verified archive restoration **before** Python package installation on an already-approved Colab T4. However, the existing H67 `restore_stage` is intentionally *create-only*: once `/content/hala-h62-work/run` exists, a **subsequent H62 launcher retry on the same surviving session** fails `FileExistsError` even if the originally staged H71 data are complete and all hashes match. This can happen if a package import, model download, or later training action failed after the first H67 admission. Repeating the 1.76-GB import by deleting the run directory would also discard valuable legitimate optimizer checkpoints.

## H81 implementation

New `h81.staged_resume.admit_stage`, called from the existing H62 pinned-source launcher, preserves H67 `restore_stage` unchanged for **first admission**. If a directory already exists, it permits reuse **only after full independent requalification**:

- exact SHA256 of original operator H71 archive and pinned **original** H71 source Git SHA (separate from H81 executable source);
- H67 ZIP integrity: immutable member inventory, stage status, correct TRAIN split, source Git SHA, safe path policy, byte sizes, known source manifest;
- **every** original restored TRAIN member re-hashed and compared to the archived SHA256/size inventory; missing, extra, altered or linked source files fail closed;
- existing H61 `validate_train(..., rederive=True)` rechecks source screenshots, TRAIN-only manifest, curriculum row and media counts and independently derives H61 dataset identity;
- allowlisted generated top-level **training outputs** may remain untouched for H61's own protected optimizer resume; unexplained run-root files/symlinks are rejected.

This is a **reverified reuse**, not a trust-on-first-use shortcut. An invalid, incomplete or cross-run source is never overwritten automatically. The original synthetic H67 restore test still refuses repeat import; H81 is the explicit verified wrapper used by H62.

Tests prove clean first restore, matching-source retry without re-extraction, retained prior training state, tampered source pixels/labels, missing or unexpected source member, wrong archive/data commit, archive alteration, unexpected workspace entry, regular file and incomplete directory rejection, and symlink guard when supported by the OS. No network, GPU or credentials used.

The H80/77 checkpoint rescue and optional private authenticated storage layers remain unchanged. The first genuine eligible free T4 and real Qwen optimizer checkpoint still have not occurred.
