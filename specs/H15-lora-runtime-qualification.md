# H15: real two-step LoRA runtime qualification

Status: planned. Depends on: H8 existing candidate helpers and actual H5/Colab runtime. Surfaces: new qualification orchestrator/tests/docs; existing trainer only if a real defect requires a bounded repair. Disjoint from H14 experimental protocol/dataset files. No large training job or adapter quality claim.

Use the existing pinned Qwen3-VL-2B, FP16, rank4, batch1/accumulation1 and 58-row English rights-cleared artifact. This smoke audits all 58 records but consumes two microbatches. The previously measured Arabic grounding deficit is not remediated by this English engineering smoke.

Acceptance: actual processor masking, forward/backward, finite nonzero adapter gradients and verified optimizer changes; immutable paired adapter/optimizer/scaler/RNG checkpoint saved; load saved adapter into the actual base model; compare uninterrupted two-step run against interruption after first atomic checkpoint followed by resumed step2 under identical identities; report tensor equality/tolerances, RNG/optimizer/state evidence, model/adapter digests, latency and allocator/device-memory measurements. If exact replay fails, preserve discrepancy and diagnose before qualifying. Completed-checkpoint finalization alone does not count as real reload/resume.

Run only on explicit remote Linux opt-in and the existing permitted free Colab T4. Pin torch2.11.0, transformers5.16.1, peft0.20.0; verify actual imported versions, not package metadata alone. No local weights, paid resources, Drive scope expansion or production serving. Remote adapter bytes stay in the notebook; save small measured metadata to Hala and label ephemeral persistence until an authorized durable remote store is available.

Builder writes meaningful CPU control-path/identity tests and a bounded source-controlled remote orchestrator. Parent performs independent PStack/Jev/Graft gates and runs the GPU job. No claims from mocks. Preserve original strict0/8 and calibration7/8 records.
