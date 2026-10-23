# H78 — Require a complete, new, gradient-bearing checkpoint before training success

**Parent:** H77 [Draft PR #61](https://github.com/TheHalfMoon/Hala/pull/61), frozen `0bc1d7c81b817e7a06662682686a669b445aa17a`.

## Source defect

The H65 supervisor previously marked a successful H62 child and any valid checkpoint egress status as `COMPLETED_WITH_HOST_EGRESS`. This did **not** independently establish that the recovered checkpoint reached the requested optimizer-step budget or even that it was produced in the current run. For example, a real step-1 checkpoint could be salvaged while the operator requested three steps; or a previously staged checkpoint might be reused without progress.

Those snapshots are valuable as partial progress and remain eligible for the separately opt-in H77 **private backup**. But they are not sufficient evidence that the entire new training job ran successfully.

## H78 fail-closed completion gate

`h78.checkpoint_evidence.qualify_completed_host_checkpoint` is a CPU-only, read-only verifier. It requires:

- Full, existing cryptographic `training.checkpoints.load_checkpoint` verification (pointer SHA256, metadata, exact adapter+optimizer file inventory, per-file SHA256, immutable stage paths and internal state).
- Checkpoint step **and** configured `max_steps` both equal the explicit H65 requested step count.
- Nonzero changed LoRA parameter tensors and a complete, ordered receipt for each requested optimizer step with a finite, positive gradient norm and a nonzero count of updated tensors.
- No reuse of the exact checkpoint SHA256 observed in host staging **before the latest supervisor launched**. Earlier staged checkpoints remain intact and can still be published privately, but cannot establish a new successful training attempt.
- A minimal, redacted result with SHA, dataset/config identities, step and evidence classification. `adapter_qualified: false`, `private_provider_readback_verified: false`, `evaluation_required: true`. This is not proof of real task improvement.

The default H65 supervisor now calls this actual verifier only if (a) the child exited zero, (b) `train` was requested and (c) at least one verified egress occurred. Failure returns `FAILED / unqualified_completed_checkpoint`, while preserving host artifacts and the complete H65 receipts. A nonzero child exit or provider interruption remains `FAILED`; H77 still may upload a **verified partial salvage** privately without rewriting the original failure outcome.

No-train H65 diagnostics retain their existing success semantics. Existing synthetic tests explicitly inject a synthetic verifier to test CLI/transport behavior, while H78's production-default verifier is exercised with immutable real-format **synthetic** checkpoint artifacts.

## Not done

No GPU was obtained, no optimizer step executed, no genuine model checkpoint produced, no provider storage publication or private readback attempted. H71 archive and Kaggle training restrictions are unchanged. The first real checkpoint still needs an authorized zero-cost T4, Qwen weight preparation, one real optimizer update, and independently verified host/provider readback followed by held-out bilingual GUI evaluation.
