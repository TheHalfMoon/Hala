# 0004. Archive small experiments on an isolated branch

Status: accepted (2026-10-15)

## Context

The original two-step Colab adapter expired; only numerical evidence and hashes survived. The founder requires verified remote persistence before a useful training pilot and a fresh-session download/load/evaluation. The existing Hala GitHub repository is authorized for writes; no paid storage, new credential grant or production model release is authorized. Base weights remain remote.

## Decision

Use the separate `experiments/small-adapter-archive` branch of TheHalfMoon/Hala for bounded small experimental adapter safetensors/configuration/manifests. Never merge it into main or the foundation PR. Enforce an 8 MiB adapter and 16 MiB bundle limit, exact pinned base/source/data/evaluation identities, safe file inventory and externally archived manifest SHA256 before loading. Exclude base weights, credentials, customer data and executable pickle snapshots. Local staging is limited to these small artifacts.

The immutable manifest-only storage probe at `ab69e61e1b08af34fb50ffc083c4330b84f4c6b8` was retrieved from a fresh free Colab T4 and matched local SHA256 `e7ed37da06b99fb422cc45cf2ec9434f02f03d6ba5d8452bd567bbb48b44e536`. It contains no trained asset. A future adapter requires separate durable hash verification and fresh-session inference evidence.

## Consequences

This reuses authorized durable storage without cloud credentials in Colab or a production release. Small binary history increases repository storage and requires explicit archival hygiene; large weights/datasets remain excluded. Revisit storage before larger artifacts. Persistence proves byte identity, never model quality or merge/deployment approval.
