# H21c independent combined review

**Changes required: one P2 telemetry/failure-evidence must-fix.** Read-only independent correctness/product/parsimony review of the semantic data bridge, shared pilot trainer, initial tests, H21/H21c spec/docs, qualified masking/scaler/checkpoint helpers and relevant H19/runtime/archive integration. No model execution, ML installation, weight download, full-suite rerun or tracked source changes by this reviewer.

## Must-fix

- **`training/scripts/train_development.py`: direct memory sampling can erase completed-update evidence and mask the original training error.** Per-step metric construction calls `memory_sample(torch)` before append/persist; a completed optimizer update can be followed by telemetry failure, losing that step's measured loss/norm/change record. The finalizer calls `memory_sample(torch)` inside `result.update` before `atomic_json`. If CUDA telemetry also fails after a device/training exception, the telemetry exception replaces the primary error and no blocked-result checkpoint is written. Independent execution of the actual finalizer AST with a fake failing sampler produces **raised=TelemetryFailed, checkpoint_calls=0, result_status=blocked**. Reuse the reviewed best-effort telemetry approach (explicit null/error metadata, no fabricated zeros), append/persist a completed update independently of memory availability, and ensure final persistence preserves the original training exception. Add tests for both paths. This is the same concrete evidence-loss pattern repaired in H21b; merely integrating the repaired runtime file does not fix the trainer's direct calls.

Executed probe: `work/h21c_combined_probes.py`, run from the task workspace using `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21c_combined_probes.py`. Its finalizer probe executes only the actual source AST finalbody with injected telemetry/persistence, never a Torch import or training body.

## Review bars

| Bar | Assessment |
|---|---|
| Data correctness | The external H19 manifest, exact deterministic reconstruction and materialized row checks precede projection. Exactly116 Mind/Type rows selected;20 Reliance rows excluded. Projected rows remove simulator state/episode metadata. Mind is explicitly text-only; Type uses SHA-bound canonical PNG paths/dimensions. Historical loader/audit unchanged. |
| Masking and modality | Full and generation-prefix processor paths use the same public user message and Type image. Existing completion helper verifies actual expanded prefix identity, nonempty completion and input budget before masking. Mind tests pass no images. Independently exercised a positive **Arabic Type** row: only public prompt+image reached processor, both processor passes used the image, and labels masked the prefix correctly. Real processor/image tokenization compatibility remains a remote gate. |
| Baseline prerequisite | Exact model/revision/runtime bytes, frozen source/data/settings, external baseline hash, package identities, measured status and unadapted zero-training markers are checked. Complete dev episode IDs and separately labelled Mind test case IDs are required; summary rederived from per-case records. Only development completion/Mind deficits permit training. Independently changed every held-out Mind diagnostic from incorrect to correct while retaining dev deficits: the gate remains unchanged. The externally archived baseline/operator verification supplies actual-run provenance; self-consistent JSON alone is not hardware attestation or independent raw-trajectory replay. |
| Persistence prerequisite | Pretraining manifest is externally hash/identity bound and must have harmless-probe kind; immutable probe-commit syntax and authorized repository recorded. Documentation correctly states that typed strings/local bytes do not prove a network roundtrip: the root operator must actually verify/download the signed immutable artifact commit before invocation. No archive credentials enter the trainer. |
| Numerical training | Source uses qualified initial-scale1 FP16 scaler, single seeded authored epoch/116 steps, FP32 adapter parameters, complete112 tensors/802816 elements, language q/v targeting, frozen nonadapter/base/vision parameters, finite loss/gradient/update gates, clipping/nonzero norm and per-step actual change. CPU contract tests do not execute these Torch forward/backward/optimizer paths; actual remote qualification remains necessary. No claim of model quality follows. |
| Checkpoint/archive | Existing atomic paired publisher validates schedule/data/manifest state and hashes; optimizer/RNG pickle stays local trusted checkpoint and is excluded from archive. Original PEFT safetensors/config are copied unchanged through the exact bounded archive validator. Final status explicitly remains pending durable archive; fresh verified path rejects harmless probes before a later PEFT load. Actual durable upload/download/load/evaluate still required before resetting the session. |
| Product | One shared adapter, no separate Mind/Type models, no test-based schedule tuning, no local weights/native execution/model authority. Actual baseline and training remain unexecuted by this source phase. Latest reviewed H21b bytes must be integrated before freezing/using the baseline; exact runtime hash requirement intentionally refuses stale baselines. |
| Parsimony | Separate small semantic bridge and fixed trainer reuse existing masking/scaler/atomic/archive helpers appropriately. Repair the evidence path locally; no generic importer, retry scheduler or new authority layer is needed. |

## Test and review state

Initial file inspection was of the seven original CPU contract tests; they do not exercise actual Torch numerics. The independent probe passed its Arabic Type input/masking and held-out-diagnostic independence checks, and reproduced the finalizer defect. During review, the builder began adding repair regressions in the shared worktree. A later targeted run therefore discovered **nine tests, with the two new helper imports failing** (`record_completed_step` and `finalize_training_result` not yet implemented at that instant). That is recorded as an in-flight red stage, not represented as a passing final suite or a separate permanent finding. Wait for the completed repair before a fresh delta review; do not freeze changing source.

Original reviewed SHA256 values, before those in-flight test edits:

```text
training/development_data.py d9a8f02b2c843276051c0220df7d8811499ed1b1d8316a9878fde32a3e288dfb
training/scripts/train_development.py ca0b9af0f42f6a9ac31c0cf2635a39a915d4cdad9ab94d53244b77ab0986ffcf
tests/test_development_training.py b10532b8d076b83996bff78fe5d20b919763ed028331cc9922c24d2918f7dbd3
```

Separate security judgment, genuine Jev/Graft, exact-source CI, signed source/artifact publication, qualified bilingual rendering and an actual baseline remain coordinating gates. This report provides no model/training/persistence/release qualification or founder merge authorization.

Panel: fresh independent combined correctness/product/parsimony; one concrete must-fix, outside-checkout probes, no source edits.
