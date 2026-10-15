# H17 Colab identity guard security delta

Verdict: no must-fix security finding in the platform-identity delta. Removing incidental directory existence as platform identity is appropriate for the parent's independently verified free Colab session; it does not authorize a Kaggle run or paid compute.

Independently verified prospective source-freeze SHA256: `d9cbb88f567e43325028151d39c1a00e2ae9c9081fdd1556094aedce8e69585c`. All listed H17 file hashes and the ten frozen H14 dependency hashes verified. The baseline, configuration, model/processor revision, generation settings and runner remain unchanged by the guard delta.

- Must-fix: none.
- Worth-considering: none newly introduced by this delta. Prior optional cache-descendant and explicit safetensors hardening observations remain outside this narrow platform correction.
- Skip-it: treating `/kaggle` existence as conclusive platform identity when observed in a positively verified Colab session. A directory name does not establish the runtime provider, billing status or commercial eligibility.

The sole executable guard change removes `or Path('/kaggle').exists()`. The guard still rejects any nonempty `KAGGLE_KERNEL_RUN_TYPE`, even when a Colab marker is also present; requires Linux, explicit `HALA_REMOTE_QUANT_ALLOWED=1` and a nonempty Colab marker; and retains external-cache containment and root-symlink rejection. The patch does not manufacture, overwrite or clear platform identity variables. It adds no infrastructure API, billing action, provisioning, telemetry, local model path or authorization bypass. Optional framework imports remain after this guard and source/media validation.

Independent bundled-Python execution of `python -m unittest tests.test_quantization -v`: 8/8 passed. The two new regression tests accept an incidental directory in an opted-in Colab-marked Linux environment and reject explicit Kaggle identity, missing Colab identity or missing opt-in. A separate source-verification process confirmed that `torch`, `transformers`, `bitsandbytes` and `PIL` remained unimported. The parent's reported 140-test suite was not repeated by this judge. No GPU run or model download was performed.

Environment markers are corroborating runtime checks, not a cryptographic identity or financial eligibility proof. This was already an operational limitation before the change: parent-side UI verification of the actual free session remains required, and manually forged or suppressed environment markers are not authorized. The parent's report of the previous startup stopping before ML imports with zero predictions supports a new prospective freeze; this review verifies the new source artifact, not independent remote execution evidence or archived run chronology.

panel: independent security delta ✓ fresh-context Codex judge (session model), static delta plus eight targeted CPU tests; no d1 approval, financial eligibility certification or GPU qualification.
