# H21c shared quality pilot

Status: source implemented, CPU-tested and independently reviewed; actual baseline, remote training and durable quality evaluation pending.

Build a separate semantic training loader and one shared rank4 language q/v LoRA pilot, preserving the historical action loader and H18 audit. Require the external H19 manifest SHA, reconstruction of authored rows/labels and media, unchanged actual H21b baseline with matching model/runtime bytes/settings/data/source, a measured development deficit, and a verified harmless archival roundtrip before model allocation.

The fixed schedule is seed42, authored row order, one epoch, batch1, accumulation1, FP16 scaler initial1.0, AdamW5e-5, rank4/alpha8/dropout0, gradient clipping1.0. Derive steps from selected authored Mind/Type rows: 116 of 136; exclude all 20 Reliance rows and all held-out episodes/Mind diagnostics. Mind is text only; Type uses its current verified PNG. No private simulator state reaches the processor. No held-out test selection, tuning, retries with changed schedules, or invented quality claims.

Acceptance: external identity mismatches, coherent label tampering, media drift, prefix-token drift and baseline/probe failures fail closed; only language adapter parameters train (112 tensors /802816 elements); nonfinite loss/gradients/updates and zero updates abort; use existing atomic checkpoint and identity helpers; archive exact original PEFT adapter bytes through reviewed bounded safetensors bundler; record losses, changes, timings, memory and identities. A fresh-session seam verifies the external manifest before returning a path for PEFT loading. The trainer does not upload or accept cloud credentials. Founder/root persistence and actual fresh-session evaluation remain required.

No local weights, GPU execution, ML package installation or paid resources are part of this implementation. Required Jev, Graft, independent PStack correctness/security review, signed DCO commits and exact-source CI are owned by the coordinating session and remain incomplete here.
