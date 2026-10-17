# 0009. Retain long GPU experiments as batch versions

Status: accepted (2026-10-17)

## Context

Multi-hour conversation pauses twice outlasted Kaggle interactive inactivity expiry. All three DEV profiles and the subsequent FP16 TEST archive were exported successfully; the later int8 TEST outcome is unknown because its VM disappeared before export. Interactive code edits did not keep that allocation available. Kaggle's Save Version dialog offers a fresh batch execution with saved outputs.

## Decision

Use a clean, independently reviewed Save & Run All notebook for long experiments. Publish a prospective retention/hardware amendment and exact operator bytes before execution. Record the actual assigned physical GPU and qualification before model calls; preserve bounded per-profile raw archives in the saved batch outputs. Verify the completed version and retrieve/hash/publish its raw evidence before claiming durable retention. Keep all prior outcomes and unknown runs distinct.

The scientific runtime, model, adapter, data, prompts and scoring remain fixed. If a full TEST triplet is replicated to obtain one physical GPU for cost comparison, declare that replication prospectively and preserve the earlier FP16 result separately. No TEST-driven tuning or automatic adoption follows. A batch submission or local ZIP alone does not prove platform durability.

## Consequences

+ A running batch can finish independently of interactive browser activity and preserve results in a versioned output artifact.
- Assigned hardware is unknown before scheduling; prospective admission and pre-inference hardware records must explicitly bind the actual allocation.
- Saved-version output retrieval and independent source/result audits remain mandatory. Failed or incomplete runs retain their evidence and stay unqualified. Free quota use and warm-cache/order confounding remain visible.
