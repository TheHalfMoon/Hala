# H21b independent security and integrity review

Verdict: PASS for the scoped runtime source; no actionable security/integrity findings. Actual GPU baseline remains unexecuted and unqualified. Reviewed H19 integration, clean published source, qualified remote media freeze and externally archived manifest digest remain prerequisites before execution.

Reviewed AGENTS.md, CLAUDE.md, PRODUCT.md, `evaluation/development_runtime.py`, `tests/test_development_runtime.py`, H21b spec/docs, plus latest Hala-bilingual freeze/runner/media interfaces. The reviewer did not author or change source. No full-suite rerun, ML package import/install, weight download or GPU inference occurred.

## Trust and allocation boundaries

- Explicit Linux/Colab/opt-in guard and Kaggle rejection precede runtime loading. Documentation correctly limits these environment markers to accidental-execution protection rather than hardware attestation.
- Expected manifest digest, frozen settings, source hashes, pinned font, expected clean Git commit and package metadata checks precede ML imports/model allocation. Actual imported versions are checked again. Independent source/manifest digests are caller trust anchors, not derived from downloaded manifests.
- Model and processor use the exact immutable revision and `trust_remote_code=False`; adapter loading always rejects before allocation. No untrusted output is executed, passed to a shell, used as permission, or interpreted as a completion receipt.
- Approved resolved remote cache paths are limited to `/content` or the documented `/tmp/hala-` prefix. Output requires a fresh directory. Generated frames and result paths are harness-owned rather than model-provided.

## Independent bounded probes

`work/h21b_security_probes.py` executed with bundled Python through `rtk proxy`, exit 0:

- Rejected altered/bool settings, model revision changes, extra settings and changed token/call/time limits.
- Rejected an existing outside-root image, oversized prompt/image, invalid PNG signature and expired deadline before model calls.
- Rejected 4,097 expanded input tokens before generation. Fixed image dimensions and bounded prompt/image bytes cap preprocessing inputs; token-limit enforcement occurs after preprocessing/device transfer, not before it.
- Preserved malformed authority-bearing prose verbatim in raw output while the callback received only the public text message. No parser repair or executable path was introduced.
- Verified nonfinite result serialization cannot replace the prior result; escaped lone surrogates persist safely in canonical JSON.
- Confirmed bundle failure occurs without importing torch/transformers/PIL, and reserved adapter input rejects before bundle/model loading.
- Confirmed an old runner missing independent Mind correctness metrics cannot obtain a summary. The latest H19 runner supplies separate semantic, planning and actual final-state metrics.

## Measurement and execution limits

Per-call token/time settings and per-episode call/action limits are explicit. Persistent monotonic checks prevent a caught alarm from starting more calls; a worker subprocess with an external process timeout is still required for hung CUDA containment, as documented. GPU allocated/reserved peaks, sampled full-device usage, process RSS and CPU/wall/model time are reported with appropriate limitations. No continuous device-memory peak, production serving cost, quality qualification or native integration is claimed.

Canonical result encoding, temporary-file write, fsync and atomic replacement preserve prior checkpoints on serialization failure. Checkpoints cannot guarantee survival of VM loss or forced termination, which is disclosed. After integrating latest H19, the prospectively frozen bundle must bind that exact source before any actual baseline output is observed.

Reviewed source SHA256:

- `evaluation/development_runtime.py`: `a68a959e5d0a213ef6d2f6bb731b1437f760486abf617d0be4ce5be9b75f3ad2`
- `tests/test_development_runtime.py`: `9dd444dcf739939b3820dafb80454460b6c2ece89917574702bae4f946e6b71e`

This is independent PStack security evidence, not actual model/GPU qualification or replacement for the remaining Jev/Graft/publication gates.
