# Two-step remote LoRA engineering qualification

This bounded English smoke measures actual forward/backward, finite nonzero
adapter gradients, per-step optimizer updates and paired snapshot reload. It is
not adapter-quality evaluation or Arabic grounding remediation. CPU control tests
are explicitly unqualified for the model runtime.

Use only the founder-approved free Colab T4 runtime with imported Torch 2.11.0,
Transformers 5.16.1 and PEFT 0.20.0. The existing trainer checks the observed
provider, Linux opt-in, dependency versions, immutable revision, baseline deficit,
rights-cleared training artifact and every image digest before allocating weights.
No local weights or paid jobs are authorized by this entrypoint.

From the repository root on that remote runtime, with the pinned dependencies
already installed and source plus the 58-row artifact present:

```sh
HALA_REMOTE_TRAINING_ALLOWED=1 python -m training.scripts.qualify_lora \
  --config experiments/runtime/colab-lora-smoke-config.json \
  --examples training/generated-candidate/examples.jsonl \
  --manifest training/generated-candidate/manifest.json \
  --baseline experiments/runtime/colab-lora-smoke-baseline.json \
  --output /tmp/hala-lora-qualification
```

The output directory must be fresh. The run consumes the first two microbatches
in two trajectories (four microbatches total); it audits all records with the
actual processor. One trajectory performs two uninterrupted optimizer steps.
The other deliberately stops immediately after the first atomic published paired
snapshot, releases its model, constructs a new base model and loads the saved
adapter/optimizer/scaler/CPU and CUDA RNG, then performs its second optimizer step.
A completed checkpoint finalization does not qualify as this reload.

`qualification-result.json` records input and source digests, checkpoint artifact
digests, actual imported packages, device/dtype parameter residency, expanded and
completion token counts, gradient norm before clipping, per-step maximum update,
allocator allocation/reservation peaks, device free/total memory and elapsed times.
`continuous/` and `resumed/` contain immutable paired checkpoints and training
results. Adapter tensor and optimizer floating comparisons use absolute tolerance
1e-6 and relative tolerance 1e-5; exact equality is separately reported. RNG and
scaler comparisons require exact equality. Shape, dtype, key, nonfinite and
out-of-tolerance differences fail qualification and leave a discrepancy report.

Artifacts remain in ephemeral notebook storage. Copying the small JSON report
into source control records measurement evidence; it does not make adapter bytes
durably available. No Drive permissions or upload are requested.

CPU checks: `rtk python -m unittest tests.test_lora_qualification -v`.
The parent engineering run supplies independent Jev/PStack/Graft gates and the
actual GPU result; passing these control tests alone establishes no GPU claim.
