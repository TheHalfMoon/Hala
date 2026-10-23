# H75 — Bounded bilingual processor audit for first-optimizer-step diagnostics

**Parent:** H74 PR #58 at `6eed2e800645a83871ae742d2ac2c2a899dc706a`.
**Status:** implemented and CPU/unit-tested. No new Colab GPU allocation, optimizer update, H61 adapter, private checkpoint or measured task improvement.

## Measured code-path bottleneck, not a speed benchmark

The H61 pinned Colab config sets `audit_records: 0`. The existing trainer interprets zero as a full processor audit: it independently calls Qwen3-VL's vision/token processor on every one of the 10,000 H71 training rows *before* first model weights are loaded and before first optimizer step. This expensive code path was inspected; **time savings have not been timed on a GPU or with the actual Qwen processor**.

H75 introduces an explicit optional first-step-only `--processor-audit-records 32` flag, forwarded exactly through H65 supervised launch -> H62 operator -> H62 remote pinned execution -> H61 runner -> config signature -> training code. The range is 1–64. It is refused for experiments longer than 3 optimizer steps, unstaged sessions, absent H67 source SHA, absent `train` step and invalid budgets. Zero/unset preserves the original *full* processor audit for regular or resumed long runs. The normal Kaggle model-training prohibition is unchanged.

The CPU sampling algorithm is deterministic, source-order stable and always takes at least one record per (language, source family) group; if the budget cannot represent all groups, it fails closed. Additional rows are selected evenly across the original rows, not just the English prefix. The processor audit receipt includes only aggregate language/family totals, never prompts or private media paths.

**All H61 source records and images still pass the original H67 full provenance/content SHA check, H61 curriculum derivation and all-record `check_media` before any optimizer step.** H75 reduces only redundant full *processor* pre-encoding during the supervised 1–3-step diagnostic. Each actual optimizer microbatch continues to run `prepare_inputs` and to SHA-check its image.

## One-step supervised command (not executed; requires genuinely existing eligible free T4)

The H71 original source ZIP must ALREADY exist at `/content/hala-h67-stage.zip` on that T4, with the independently verified complete ZIP SHA. Running-code SHA must be the eventual reviewed H75 commit, **not** the original H71 source SHA.

```bash
python -m h65.safe_diagnostic \
  --session EXISTING_AUTHORIZED_FREE_T4 \
  --pinned-sha EXACT_REVIEWED_H75_CODE_SHA \
  --destination PRIVATE_HOST_RECEIPTS \
  --steps train \
  --stage-sha256 4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a \
  --stage-source-sha acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe \
  --max-steps 1 \
  --processor-audit-records 32
```

H65 still requires an explicitly allocated **free** T4, source SHA validation, independently verified checkpoint staging, and truthful egress. This flag never downloads/mounts the H71 dataset, acquires a GPU, skips model/runtime verification or declares success. If a real first checkpoint is obtained, separately qualify its training identity and private storage before any full run.

## Code-first REA and potential next blockers

- Hugging Face pinned offline snapshots: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- Snapshot weight-cache completeness is version-sensitive (e.g. a cached file inventory can be incomplete): https://github.com/huggingface/huggingface_hub/issues/5112
- This H75 optimization does **not** eliminate model weight downloading, processor/model loading, H67 restore overhead or T4 reclamation. H74 cache candidate is not proof of a warm GPU model.
- Reconfirm actual Colab capacity, permission and zero cost. Do not churn allocations after known failure.
- First durable target remains one **actual changed gradient-bearing parameter + immutable checkpoint + verified host readback**, then held-out English/Arabic GUI outcomes. Neither is proven here.

## Evidence

Unit tests are synthetic and provider-free. Tests assert group-balanced deterministic sampling, failure for insufficient coverage, default full audit, host CLI propagation, remote fixed-source verification, fail-closed bounds, actual H61 config override and source-config immutability. They cannot prove actual speedup or model quality.
