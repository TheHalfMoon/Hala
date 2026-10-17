# H21a independent combined review

Verdict: changes required before qualifying this archive as the persistence path
for the actual qualified PEFT producer. Read-only review; no source edits.

## P1: actual PEFT 0.20 producer configs are rejected

`training/adapter_archive.py:105-114` rejects two ordinary properties of the
actual qualified producer. `training/scripts/train_adapter.py:24,154-156` passes
the exact language-only regex `.*language_model.*\.(q_proj|v_proj)$` as
`target_modules`; PEFT supports and preserves a string here, while the archive
accepts only the two-name list. A matched fixture changing only that field to
the actual source constant is rejected with "Only pinned rank4 language q/v LoRA
accepted". Thus the real saved adapter cannot be archived unchanged.

The exact official PEFT 0.20.0 source also has serialized dataclass fields
`peft_version`, `lora_ga_config`, `monteclora_config`, `use_bdlora`, and
`velora_config` missing from the archive's field allowlist at lines21-35. Its
`to_dict` returns `asdict(self)`; it fills `peft_version` automatically. The four
variant fields default to `None` (including `use_bdlora`, which is a config-valued
field, not a boolean default). Even after fixing the target regex, a normal
PEFT0.20 config still fails "Unreviewed adapter configuration fields".

Minimal repair: accept the exact qualified language-only regex in addition to
the existing list form; do not allow arbitrary regexes. Add the exact 0.20 fields,
require the recorded supported PEFT version and reject non-None variants. Add a
realistic producer-config fixture using the complete official default config,
with regression cases for a broadened regex, unknown version and enabled variant.
Preserve producer bytes; do not strip fields or broaden the qualified training
target to make archival tests pass. A remote real-config archive/load seam check
remains necessary before training-pilot persistence is considered ready.

## Evidence and assessed scope

- Ran the nine existing archive tests: passed, with the real Windows symlink
  creation case skipped because the account lacks privilege (0.553 seconds).
- Independent probe confirmed actual target regex rejection, malformed immutable
  identity rejection, and successful CLI creation using absolute paths with spaces
  from an unrelated working directory.
- Inspected exact official wheel source in memory, without importing/installing
  PEFT, Torch or model weights. Wheel URL and SHA256 match PyPI metadata:
  `peft-0.20.0-py3-none-any.whl`, 775,777 bytes,
  `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233`.
- Reproduction script: `work/h21a_combined_probe.py`; complete results:
  `work/h21a-combined-probe.json`. Both are outside the source worktree.
- Fixed finite file/header/payload budgets, canonical external manifest trust
  anchors, duplicate-key rejection, contiguous offset accounting, no pickle/ML
  imports and an explicit no-quality claim are proportionate to this subphase.
  Tensor finiteness, all-layer coverage, exact Qwen projection shapes and real
  loader compatibility are deliberately not proven by the offline format checks;
  the documentation makes this limit explicit. No production readiness inferred.
- No full-suite rerun, GPU task, archive publication, model download, installed
  source edit, deployment or authority grant occurred in this review.

The separate security judge remains required. This review is correctness,
parsimony and product-fit evidence and cannot substitute for that gate.
