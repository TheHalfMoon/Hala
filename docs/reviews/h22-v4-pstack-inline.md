# H22 V4 — PStack /ps-review degraded local panel (2026-10-17)

**Manifest: panel: light — correctness inline; parsimony inline; product inline; security inline. Fresh-context judge unavailable in this session. This is NOT independent PStack reviewer signoff or Hala release qualification.**

Review scope: from H22 V3 `b9ac75188025145247fd5ae6a3bc1be976a7a09a` to the proposed V4 delta in `tools/generate_h22_batch.py`, `tools/h22_batch_operator.py`, `tools/verify_h22_batch.py`, generated Kaggle notebook, tests and the prospective log.

- **Correctness:** V3 measured FP16 blocked on `torchao 0.10.0` ImportError during PEFT 0.20.0 LoRA attachment. The V4 notebook uninstalls that *optional* package in its fresh remote runtime before any model allocation; both notebook and batch operator fail closed if the incompatible distribution still appears. This does not prove the next remote FP16 attachment succeeds. The full TEST batch must still be retried on a new allocation.
- **Parsimony:** The existing pipeline and fixed historical model/data are reused. One package removal and two explicit environment assertions are simpler than version-upgrading an unused optional backend or changing scientific source files.
- **Product:** Correctly reports partial V3 evidence and preserves earlier `UNKNOWN` interactive INT8, original V3 blocked FP16, and no confidence in quantization adoption or commercial serving. No new user-facing features.
- **Security/operations:** No credentials, public tunnels, privileged local execution or paid infrastructure introduced. The only new package command removes preinstalled `torchao`; the separate pinned notebook environment and original frozen model hashes remain unchanged. Remote free-tier terms and session permission must be checked at submission time.
- **Actual tests:** 20 targeted tests, 275 full CPU tests (one skipped), genuine Graft build/check, genuine Jev five-file advisory judgments. The real GPU rerun and post-retrieval ZIP validation remain mandatory.
- **Blocking gap:** Fresh-context separate reviewer unavailable through the current tool harness; do not mark this panel independent. Jev correctness-concern advisory estimates are not definitive approval or concrete explanations.

Disposition: **NO MERGE / NO ADOPTION / CONTINUE QUALIFICATION.** The founder retains merge and release authority.