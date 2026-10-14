# H14 + H15 independent security delta review

Verdict: no must-fix security finding in the reviewed experimental changes. This is source/control-test evidence only; actual GPU evaluation, LoRA qualification, package-install provenance and the parent Jev/Graft gates remain separate.

Reviewed on 2026-10-14 as a fresh-context PStack security judge. Scope: new H14 output parsing, source/image freeze and remote evaluation; H15 deliberate partial interruption, resume comparison and runtime evidence. Existing frontier security review was not rerun. Read both repository AGENTS.md files, Hala CLAUDE.md, active specs, the supplied diffs/inventories and narrowly relevant pre-existing artifact/checkpoint/preflight helpers. No SECURITY.md was found; repository rules remain authoritative.

## Findings

- **must-fix:** none.
- **worth-considering:** none required for the present operator-controlled experiment boundary.
- **skip-it:** `evaluation/heldout_fixtures.py:146,172` — fixture validation is not a general hostile-upload schema, and manifest file membership is operator-authored. The externally recorded manifest digest is the trust anchor; generated cases have fixed safe IDs and frozen source hashes. Do not treat an attacker-selected manifest plus its own digest as a trusted freeze. No change required for this boundary; add an exact bounded schema/file allowlist before accepting third-party manifests.
- **skip-it:** sibling `training/scripts/qualify_lora.py:74` — `snapshot_state` reads checkpoint metadata before the stricter `load_checkpoint` pointer/path check. Current outputs are fresh, local experiment directories produced by the trainer, with no external checkpoint intake. No exploitable new attacker path established. If checkpoint import becomes supported, call a shared validated pointer resolver before any metadata read rather than constructing the path from the pointer directly.

## Boundary evidence

- `evaluation/heldout_protocol.py:16,50`: model text has a 4096-character cap; only the declared exact JSON fence is unwrapped. Duplicate keys, nonfinite constants, extra fields, booleans, floats and out-of-range coordinates fail. Integer dimensions and half-open pixel geometry bound the resulting point. Model assertions cannot create completion; scoring inspects a synthetic selection transition. Prompt construction allowlists goal plus protocol, excluding private IDs, boxes and fixture state. Visible UI labels carry task information without expected-target markers.
- `evaluation/heldout_fixtures.py:172` and `heldout_runner.py`: trusted manifest hashes bind reviewed sources/tasks; resolved manifest paths must remain under ROOT. All images, font and rendering provenance are persisted before model loading, then image bytes are rechecked per call. Fixture IDs used in PNG names are locally generated frozen values, not model outputs. Failed/partial runs never receive the full-run aggregate. Remote Linux opt-in and Kaggle rejection precede loading; the pinned model uses `trust_remote_code=False`. Generation is capped at 48 tokens with a 15-second stopping criterion and a 1800-second process alarm. Documentation accurately states that a stalled GPU kernel need not be immediately interruptible.
- Sibling `training/scripts/train_adapter.py:76,127,170`: remote/Linux, platform, configuration, baseline, rights/artifact and CUDA preflight precede model weights. Imported Torch/Transformers/PEFT versions are checked, beyond installed metadata. These version checks are consistency evidence, not wheel-origin attestations. Pinned dependency acquisition is parent/operator work; no install, cloud provisioning or billing call is introduced here.
- Sibling `training/scripts/qualify_lora.py:125`: exact two-step/rank4/batch1/accumulation1 configuration bounds the experiment; input digests are checked between trajectories. A real partial step-one snapshot, deliberate interruption and reload from step one are required. Completed finalization cannot stand in for reload. Injected CPU trainers always produce `cpu_control_test_unqualified` with `runtime_qualified=False`; quality and Arabic remediation flags remain false.
- Checkpoints are published as atomic paired versions with file-set/content checks and constrained pointer paths. Adapter writes use safe serialization; comparisons load adapter tensors with safetensors and optimizer/RNG snapshots with explicit `torch.load(..., weights_only=True)`. Comparison checks identities, structure, dtype, nonfinite values and tolerances, with strict RNG/scaler comparison. There is no new shell/eval execution from model text, production authority, customer secret collection, or security-policy override.

## Independent probes and limits

Bundled Python, through RTK, ran `python -m unittest tests.test_heldout -q`: **14 passed**; sibling `python -m unittest tests.test_lora_qualification -q`: **6 passed**. These cover malformed/fenced outputs, coordinate edges, private annotation isolation, false completion, freeze tampering, local-runtime rejection, interruption placement, identity mutation, comparison failure and CPU unqualification. Actual tensor comparison/deserialization and GPU behavior were not executed because this local runtime has no ML packages; local rendering tests used Pillow 12.3, not the pinned remote build. No weights, downloads, GPU jobs, deployments, commits or source edits were performed by this judge.

Both immutable diff SHA-256 values were recomputed and matched:

- H14: `5c549e2d1e1b8c080a8fcef20033cd604241ca0a047b0b735d7c08cea3f152c9`.
- H15: `b8aea10edd4a80d74784093ea075cd67802db42fbab2bd5b65790a224ee496b8`.

All **13 source/doc/test inventory hashes** matched `h14-grain.json` and `h15-grain.json` at review time. Principal source hashes: H14 protocol `d040066be79faf5521f2f227484b0a0eeb41810889004755dd1be5a4801d03a9`, fixtures `3458d012b8d90582e2528e9b08be928e393b925ff9f513fbef09de8b76ee8aae`, runner `80518bf1201ad219dfa5b8330f6f9a1955aad9ae1de8cceb5ceb38f7629f4310`; H15 trainer `b04afed6cdd64c252694b03375ddb4600e2772e36568576fbe53e183b88e1983`, qualifier `87638bd2ac350c6a612f40f117724ee9975d5242617538ef30ccbc33f2583317`.

Manifest: **panel: full — security ✓ separate fresh-context Codex PStack judge (session model, inline security bar); targeted CPU tests ✓; immutable inventories ✓; d1 not used; real GPU/package-origin/parent Jev/Graft gates not supplied by this review.** No merge, deployment or runtime qualification approval is implied.
