# H318 — Cross-platform asset custody identity guard

**Research-only metadata hygiene. No data admission.** H318 extends H317's
metadata-only, fail-closed parser with conservative cross-platform
name-collision checks.

- Reject distinct rows whose source repository and asset path differ only by
  letter case, or whose asset paths map to the same Unicode NFKC/casefold
  representation. Source commit pins remain mandatory and unchanged.
- Require NFC-form Unicode asset identifiers; conservative Unicode
  compatibility alias collisions are refused rather than silently normalized.
- Reject reserved Windows basename/device names (`CON`, `PRN`, `AUX`,
  `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`), including extensions and
  superscript-digit compatibility aliases. Arabic filenames are still allowed.
- Keep the H316 exact SHA256 content and rendered-frame duplication checks;
  these are **untrusted supplied metadata digests**, not independent media
  verification.

No files are opened using asset paths. No image, model output, screenshot,
private data, or training material is processed. Conservative rules may
reject legitimate distinct paths on a case-sensitive filesystem; blocked
items require independent review and are never automatically merged or
admitted. The H199 approved TRAIN manifest remains empty; H173 is sealed,
H250/H251 remain quarantined, and H293/H294 safety blocks are unchanged.

Local synthetic-only regression command:

```sh
python -m unittest discover -s tests -p 'test_h318_*.py' -v
```

All H315/H316/H317 compatibility tests should also pass. No trained-model
gain, legal rights clearance, new OOD score, GPU experiment, release, or
publication is asserted by H318.
