# H76 — Host Checkpoint Crash Recovery

Parent: H75 PR #59, SHA `3779e5b0a939f8db97d8871a7232120961ea622e`.

**CPU-only recovery implementation and synthetic tests.** No new GPU usage, optimizer step or real adapter.

## Recovery window

The host-side H65 egress previously rejected an already-present checkpoint directory if the process was interrupted **after** the SHA-verified immutable directory rename **but before** publishing `latest.json`. In that case, the complete recovered adapter and optimizer files existed, but the next attempt rejected them with `target version already exists unexpectedly`.

H76 reads the remote `latest.json` pointer and its checkpoint metadata, then revalidates a preexisting final version through:

1. Exact remote-pointer SHA256 for its local checkpoint manifest.
2. All artifact SHA256s, expected file set, state and step with the existing `validate_version` verifier.
3. Exact state and artifact manifest equality with the remote metadata.
4. Non-symlink destination path and the independent `load_checkpoint` verifier after atomically re-publishing the pointer.

A passing restart produces `RECOVERED_HOST_STAGING` and downloads **zero additional adapter/optimizer bytes**; metadata are still re-fetched. Both the H65 CLI and supervisor accept this verified recovery status. Corrupt/partial/extra files remain blocked with no success pointer.

## Training-run anti-fork

If the host has an existing verified checkpoint, every newer remote checkpoint must preserve all three cryptographic input identities (`config_sha256`, `dataset_sha256`, `manifest_sha256`) **and** `max_steps`. Different experiments cannot silently replace checkpoint history even if the same Colab filesystem path is reused. Identity-preserving later steps remain supported.

## Evidence / limitations

All tests use synthetic CPU adapter/resume bytes, not real Qwen3-VL weights or a Colab session. They exercise both the empty-pointer and stale-pointer crash windows, corruption, unexpected files, divergent remote training dataset identity, compatible subsequent steps, and supervisor recognition.

**Host recovery is not durable private provider storage.** Do not publish or release an adapter based on this code. The original H71 provenance, Kaggle training prohibition, and H75 bounded diagnostic policy remain unchanged. Actual free T4 availability and first gradient-bearing optimizer checkpoint are still unverified.
