# H17 independent Colab guard delta review

**PASS for the minimal host-guard repair.** The only helper change removes `/kaggle` directory existence as a provider discriminator. Linux, explicit opt-in and a Colab release marker remain mandatory; an explicit Kaggle execution marker still rejects the host. An incidental directory is not reliable provider evidence. External-cache and cache-symlink rejection are unchanged. Root must still verify that the actual allocation is free Colab T4, as the existing protocol requires; environment markers alone are not provider attestation.

Verified archived freeze `source-freeze-before-colab-guard-repair.json` is byte-preserved at SHA256 `e35342f0e6f58b53c25ee7220ca5ee25484673d718096791e7680823ff975d46`. The real source verifier accepts new freeze `d9cbb88f567e43325028151d39c1a00e2ae9c9081fdd1556094aedce8e69585c`. Only `evaluation/quantization.py` and `tests/test_quantization.py` hashes differ in the frozen file map; all eight other frozen H17 files and all ten H14 dependencies verify unchanged. Model, dataset, prompt, generation, processor, scoring and quantizer settings are unchanged.

Inspected exact source delta against the previous preparation checkout: one guard line changed; two regression tests added. Independently ran `python -m unittest tests.test_quantization -v`: **8 tests passed in 0.061 seconds**. New coverage allows Colab with an incidental directory and rejects explicit Kaggle, absent Colab marker and absent opt-in. The caller reports a preceding RED regression and a full-suite run; this judge independently confirms the phase GREEN only.

- Must-fix: none in this delta.
- Worth-considering: preserve the failed pre-import startup attempt and provider probe as separate provenance. A new prospective freeze must not erase the distinction between a startup attempt and actual quantized inference.
- Skip-it: changing model pins, prompt, data, scoring, generation or numerical settings to work around this provider-identification failure is unwarranted.

The caller reports the first int8 startup failed in the host guard before ML imports and produced zero predictions; this judge did not inspect the remote UI or independently observe that event. The new freeze declares zero model runs and `before_quantized_model_runs`, consistent with zero inference but not a claim that no startup was attempted. GPU compatibility, quantized quality, equality/adoption, merge and release remain unqualified. No model execution or source mutation was performed by this judge.

panel: light combined correctness/parsimony/product delta (inherited session model); separate security and mechanical evidence remain the caller's responsibility.
