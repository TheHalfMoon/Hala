# Hala H25 first prospective DEV baseline — limited engineering review

2026-10-18. **Hala repository only.** This is a source/audit review, NOT actual H25 GPU-model evaluation, production deployment, or fresh-context independent PStack security approval.

## Real review evidence

- New `h25/baseline.py`: only 42 prospective, public H25 DEV examples, 21 Arabic + 21 English, from three new families. The actual LLM prompt includes **only** validated `model_input` data and a fixed, versioned seven-pair policy. `source_ref`, oracle labels and state are held outside the prompt for CPU scoring; corrupted/unexpected outputs fail closed and cannot execute device operations.
- Immutable H25 original authored source SHA256 `b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b` and model-only DEV projection SHA256 `cceed2575f3f01995d489c811e2ee38f32272492fc2b533c0197ec0ba27c8ccd`. H22/H24 TEST and original measured H24 V1/V2 outputs remain untouched.
- Strict opt-in for exact SHA40/clean-Git **remote Kaggle Batch/Tesla T4** and pinned dependencies; no host-side model download or local external execution in CPU suite. Neither environment assertions nor a downloaded file proves external GPU hardware or usage billing status.
- `tools/verify_h25_dev.py` requires independently recorded SHA256 of a downloaded file, strict model/source/protocol identity, all original 42 DEV rows in correct order and raw-answer replay; rejects missing/falsified summary and claims of trained/deployed model. **It is not an independent implementation of semantic labeling**: the source oracle and scorer are shared and tested with explicit synthetic fixtures.
- Source-only QA includes 21 new CPU tests. Fixture name explicitly identifies fabricated oracle inputs, NOT actual model predictions or observed Hala accuracy.
- **Full 368 Python tests run (367 passed, 1 skipped) before the final fixture-label-only change.** Rerun exact-head complete suite before publication; do not claim it until executed.
- Genuine Graft v0.21.1 `build` succeeded with 1,198 source-wiring nodes / 126 files and `check` reported `graph check: OK`; **deep meaning tier not built**. A first log-capture attempt failed for the wrapper's PowerShell stderr handling; direct CLI retry succeeded. No false pass claims from that initial error.
- Genuine Jev v2026.919.0 / jev-1.13.0 on six exact files completed with `ok:true` advisory questions. It flagged a higher advisory probability about confusing **synthetic fixture accuracy with actual model performance** in the test file; fixed the fixture's naming/docstring, tests rerun, and final source should be re-reviewed. Jev is not a full security/semantic review.
- Independent bilingual expert review, new unseen/test families, real HalaType/RAQM media, H23 CUDA cancel/restart and native Orcel qualification are **NOT complete**.
- Alibaba local code review and true fresh-context PStack are not qualified by this document; no claim that they passed.

## QwenGyre relevance

The accompanying ADR reviews Alibaba/Qwen's newly announced RL architecture (arXiv:2609.33848). Its trajectory-prefix deduplication, durable recovery and partial-work progress verification can inspire **future** Hala data provenance; elastic multi-node scheduler and Qwen3.8 2.4T training are not viable evidence or requirements for one free T4. No official publicly licensed QwenGyre source was verified, and no QwenGyre code was imported/copied.

**Verdict:** Source research work may be published into a signed, DCO, Draft Hala PR stacked on #7 if remaining exact-head tests and review notes pass. No training, Kaggle push, PR merge, model adoption, release, Orcel changes, paid GPU or local big weights.
