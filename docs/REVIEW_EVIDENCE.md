# Review evidence

2026-10-14. These gates cover the implemented CPU source and candidate training helpers. They do not qualify a foundation model, adapter, numerical resume equivalence, native Aladdin integration or release.

- CPU: 91 tests passed on Python 3.12.14; syntax, 35 pinned metadata hashes, whitespace and all 58 English SFT records/media passed. Commands and hashed logs: `reviews/cpu-gates-final.json`. The Linux mock container separately passed 20 transport tests.
- PStack: independent combined, security and evaluation reviews ran. Evaluation delta resolved mutable-provider, source-binding, budget and duplicate defects. Foundation repair delta 2 resolved media integrity, atomic paired snapshots, nonfinite cost results and strict diagnostic JSON defects. See the individual reports under `reviews/`; they preserve snapshot hashes and qualification limits.
- Jev: 21 genuine source judgments, no request errors, with exact source hashes and typed responses in `reviews/jev-source-judgments.json`. Scores are advisory; they do not replace tests or independent review.
- Graft 0.21.1: actual build, check and blast-radius commands returned zero; 413-node source graph. Logs and summary are under `reviews/graft-*`. Deep context was absent and informational; no managed or sandbox execution is claimed.
- Alibaba OCR: actual delegated preview/rule commands were used. No managed model review is claimed.

Reproduction: `python tools/qualify_cpu.py`, `python tools/review_jev.py`, and PowerShell `tools/qualify_graft.ps1`. Jev requires existing account access; its responses disclose token usage, not verified account charges.

The initial training builder hit an account usage limit. The later independent training and final cost delta reviews completed successfully. The generic Kaggle tensor probe is runtime evidence only; terms prevent commercial Hala model work there. Colab later assigned a free T4 and actual unchanged FP16 multimodal inference completed. Strict 0/8 and explicit normalized calibration 7/8 are preserved separately; no training or quantization result exists yet. Provider/font and coordinate delta evidence is separately recorded.

Colab delta: provider/font setup and explicit coordinate calibration passed independent review. All saved decoded points and hits recompute; strict 0/8 remains immutable, normalized 7/8 is same-fixture calibration. Fresh narrow Jev source hashes are in `reviews/jev-colab-provider.json`; Graft latest graph is 413 nodes. Neither score nor source review qualifies complete-task or adapter quality.

## 2026-10-15 verified continuation

H14/H15/H16/H17/H18 source and repair reports are under docs/reviews. Source integration153 tests passed; four GitHub jobs passed at5b8c63c. Independent H17 verification2048 assertions covers all192 raw FP16/int8/NF4 outputs and logged measurements. H18 coherent substitution/false-evidence findings were repaired with pinned historical inputs and13 phase tests. See CURRENT_STATUS.md for exact source commits, memory metrics, limitations and historical DCO attestation. Founder release/merge remains gated.
