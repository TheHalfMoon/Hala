# Kaggle non-commercial research admission - 2026-10-19

Status: private, bounded hardware admission probe completed under the user's explicit non-commercial research scope and existing account free quota. Published private kernel: https://www.kaggle.com/code/abdulazizshehri/hala-noncommercial-gpu-admission-20261019.

The submitted probe was bound to Hala source commit 0d51fb68830c7011726f4e1b70aabbe0231d64d4 and SHA256 dd25d7c1d025db0cff60c80bbab2afb1bd95119083bb605b61e07c95e7c50135. Kaggle's published source SHA256 is bc309a3f07496a6edfc8e88d5b7e45594e63cc63939ae73d921ff49ba1cc1ce5; the captured source verification reports equality after universal newline normalization (the published copy had 77 CRLF line endings). The kernel metadata confirms private execution, GPU enabled, internet disabled, and no model/data sources.

The runtime receipt reports two Tesla T4 devices, each with 15,636,037,632 bytes VRAM, CUDA 12.8, torch 2.11.0+cu128, transformers 5.16.1, and accelerate 1.14.0. The measured operation was four small FP16 tensor backward steps per device. No model weights were downloaded; no model training checkpoint or model inference was run. This is hardware/runtime admission evidence only. It does not qualify model training, inference, task quality, accuracy, commercial use, or release.

Quota receipts show GPU usage displayed as 0.90h of 30h before and after, with 29.10h remaining and a reset at 2026-10-24T00:00:00. The service's rounded display does not prove that the probe consumed zero quota. No paid compute, billing change, internet access, or quota bypass was used. Further use remains bounded by the user's authorization for non-commercial research within existing free quotas.

## Copied receipt hashes

All copied JSON bytes are preserved from `C:/Users/Shehr/.codex/codex-background/briefs/hala-kaggle-noncommercial-probe-20261019/`. SHA256:

- admission.json — 50df4f669f5c490fee840579fec45e5047207188be3e376cedad04c856eda4bb
- quota-before.json — 9b8e0eb4cc5fc9f449d621efff589cbe8c8abbd967e793528a9dd8d7ca21bb00
- quota-after.json — f2ca178242a51f4a878c98829bf1b1b259e488463279718494d247b4b90a9e5e
- research-admission-assessment.json — 451a726da9ba999de3e350babfbfe61774b0f09e14a751ac5e73c16a11afe0af
- source-verification.json — a47c72284f0f3c69f79259ed66526323977d8533b456ce97b3eed01c3515ccf3
- runtime-probe.json — 2ec50e7e23fcdb158fa0778bf5d77901c16b293aaca31d9dd5a7e1bc282ef09a
- published-kernel-metadata.json — 50a3a8ea82b446f3644a863f4ee0502f8be6ef99770613bbce920968d2535e76

The raw receipts are retained here in compact form. gpu_probe.py, logs, notebooks, and generated HTML were intentionally not copied. The broader engineering ledger links this admission to the current Hala handoff without changing historical experiment records.
