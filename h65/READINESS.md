# H65 — Free GPU Bootstrapping and Durable Checkpoint Egress

**Status:** implementation and deterministic tests. **H61 model outcome:** UNMEASURED.
**Source:** H64 Draft PR #48 head `a9d43081820cb36473575d94dabd3e11bbe8403d`.
**Branch:** `ps/h65-free-gpu-egress-and-bootstrap`. **Release:** NO-GO.

## What actually happened on the approved free Colab T4

On **2026-10-23**, `hala-h64` was genuinely allocated as a T4 and the H64 pinned source was launched. The source SHA matched `a9d43081820cb36473575d94dabd3e11bbe8403d`. One early H61 run receipt had `steps=export,firstparty,curriculum,baseline,train,evaluate` but **no completed steps**. The provider later pruned the session and reported **zero active assignments**.

The original operator-side `colab exec` remained blocked after Colab removed its session. The stale local child was terminated by SIGTERM after the provider independently showed no session, and the operator wrote an explicit **failed** receipt: remote command exit `-15` after approximately 1,413 seconds, not a completed model experiment. Its captured output ends at:

```text
H62_INSTALL ['transformers==5.16.1', 'peft==0.20.0']
```

There is **no evidence that the model trained, that any optimizer step ran, or that a checkpoint was produced**. The provider's reason for stopping the GPU remains unknown; do not assert that package installation alone caused the disconnect.

## CPU preflight — measured, not assumed

A separate approved Colab CPU session (`hala-h65-cpu`) exposed these actual installed versions:

| Package | CPU Colab installed | H61 requirement |
| --- | --- | --- |
| torch | 2.11.0+cpu | 2.11.0 |
| transformers | 5.20.1 | 5.16.1 |
| peft | 0.21.2 | 0.20.0 |
| Pillow | 11.3.0 | 11.3.0 |

In that free CPU session, a pin-only installation with
`python -m pip install --disable-pip-version-check --quiet --no-deps transformers==5.16.1 peft==0.20.0`
**completed successfully in 17.327 seconds**, and imports of `AutoProcessor`, `Qwen3VLForConditionalGeneration`, `LoraConfig` and `PeftModel` all passed.

H65 makes that bounded install the only automatic package-changing action on Colab. It rechecks every exact version and import before model weights are downloaded. A missing transitive dependency is a **blocked experiment**, not permission for an unbounded dependency reinstall. This CPU result does not guarantee the same timing or correctness on another GPU instance.

The CPU test session was explicitly terminated. Do not leave free Colab sessions running after experiments.

## Host checkpoint egress and private publication

The training code creates immutable, local-to-Colab `latest.json` pointers plus adapter and optimizer/RNG artifacts, but the previous operator did **not** actually transfer or verify the full bundle before session loss. H65 adds three distinct operations:

1. `python -m h65.checkpoint_egress --session <your-existing-session> --destination <host-recovery-dir>`: read only; downloads remote pointer, metadata and individually identified artifacts using `colab download`. Verifies SHA-256, source checkpoint metadata, exact artifact inventory, step monotonicity and the project's existing `load_checkpoint` validator; refuses missing, corrupt, non-resumable, forked or rolled-back versions. A successful result is `VERIFIED_HOST_STAGING`, **not** provider-publication evidence.
2. `python -m h65.safe_diagnostic --session <existing-T4-session> --pinned-sha <reviewed-40-char-SHA> --destination <host-dir>`: starts the H62 diagnostic **only inside a T4 the provider has already granted**, polls for verified checkpoints every 90 seconds, checks provider session presence, and terminates an orphaned operator-side command after three consecutive missing session polls. Even a successful H62 run is refused as successful training when no verified host checkpoint was recovered. Does **not** acquire T4s or enable paid resources.
3. `python -m h65.provider_publish --checkpoint-dir <host-dir>/checkpoint-staging --work <persistent-receipt-dir> --delete-host-after-readback`: OPTIONAL, gated by an explicitly configured `KAGGLE_API_TOKEN`. Uses the existing H61 private Kaggle publisher under the owner account, with authenticated readback and anonymous HTTP 403 verification. Host cleanup occurs **only after** successful private byte-level readback. If the provider fails or is not configured, keep the host recovery staging intact, report `NOT PUBLISHED`, and do not silently delete the sole copy.

No credentials, training screenshots, checkpoint binaries or provider secrets are committed to GitHub. No publication to an external dataset is claimed by the H65 code or tests. Store the nonsecret receipt JSON in GitHub after independent verification.

## Remaining before the next T4

- **Data staging:** the current H61 default still does original HTTP-range export and first-party browser rendering inside the GPU allocation. That CPU-heavy work must be made durable and transferable before another long training attempt. It is not fixed in H65.
- **Real end-to-end rehearsal:** H65's snapshot tests use synthetic checkpoints and provider stubs, not a real live GPU checkpoint. A genuine separate-inference H63 paired-visual test also remains outstanding.
- **Private storage:** publishing requires functioning owner credentials and demonstrated private readback. Only then can temporary host copies be removed.
- **Capacity:** the provider can revoke free GPU sessions at any moment; actual T4 grant, training success, visual grounding and competitive performance are never inferred from CI.
- **Cost guard:** no Kaggle commercial GPU training; private Kaggle storage and authorized free Colab compute remain separate decisions.

**Operational next:** verify current provider sessions and quotas; complete CPU preparation and private data staging; then launch **one** approved free T4 through the H65 supervised operator, publish SHA-verified checkpoint privately, and run H60 vs H61 matched evaluations. Keep PRs as Draft; no public model release.
