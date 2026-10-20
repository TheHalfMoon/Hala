# H47 original paired prompt private Kaggle bundle — immutable prerequisite

**Source signed pre-registration:** [H47 Draft PR #32](https://github.com/TheHalfMoon/Hala/pull/32), exact source `abbdc218e8562e800b6335afc71ff60af8f5ba81`. Parent is real H46 eight-model-attempt **0/8** provider-negative source/evidence SHA `34756b41b8219200b9a14af3f5e717435e83e443` in [PR #31](https://github.com/TheHalfMoon/Hala/pull/31). This H47 archival bundle is **NOT** evidence of model inference or GPU success, merely proof of what source will be submitted once all admission gates pass.

Notebook archive (byte-verified unchanged from clean source generator):
- `original-bundle/hala-h47-paired-original-vs-no-number.ipynb` SHA256 `5763ae7c489eed9510f52aaa0a3a17a245d6535ffed7ab2cc228ca3a8abdbfbd`.
- `original-bundle/kernel-metadata.json` SHA256 `bf86b8348d694cd8ffe3689cc0a782a05e14946c4b7959eecff0761f90e97994`.
- Intended private Kaggle kernel ID `abdulazizshehri/hala-h47-paired-prompt-dev-eight`. Kaggle may canonicalize URL by title; if so preserve original metadata and record actual provider URL separately rather than mutating the already pinned bundle.
- Private free Kaggle T4 only (explicit opt-in), no paid fallback; precisely eight previously authored public DEV first-frame H44 screenshots (4 EN/4 AR); 16 total actual candidate-model generations, two arms exactly matching earlier H36 `NUMERIC_EXAMPLE` and `NO_NUMERIC_EXAMPLE` prompts, two arms per task and balanced order. The numeric arm is byte-identical to H46's original prompt.
- Original model frozen Qwen3-VL-2B Instruct FP16/checkpoint revision; same per-trial decoding budget 96 max tokens, deterministic, same free T4, no synthetic fake model outputs inserted, no native UI click. Independent original H44 evaluator-only oracle must be used **only after** actual raw journal download; no direct self-grading and no success-only filtering.
- **Not an independent blind test**, external real-app or OSWorld/ScreenSpot comparison, and does not qualify model/production despite any potential improvement. Preserve all negative attempts. No separate H41 experiment or other repo changes.

Before provider submission: exact signed source SHA and notebook bytes, upstream source PR exact-head Python 3.11/3.12 GitHub CI PASS, this original archive in signed GitHub, authenticated Kaggle free GPU quota and absence of prior H47 notebook, then one private kernel version 1 only. Retain full terminal provider evidence, raw SHA, CPU external pairwise replay and correct per-language denominators even if all sixteen fail. Do not resubmit in response to errors without a new separately authorized protocol.

## Provider accepted original version 1 despite CLI client error

Original single `kaggle kernels push -p` produced a **client-side JSON decoding error** (`Expecting value: line 1 column 1 (char 0)`, exit 1 after 152.36s). The first client failure has been preserved, but is **not evidence that Kaggle rejected the submission**.

Subsequent authenticated read-only provider checks revealed the exact intended slug in the user's Kaggle notebook list page 2 and `kaggle kernels status abdulazizshehri/hala-h47-paired-prompt-dev-eight` returned **`KernelWorkerStatus.QUEUED`** exit 0. The original provider response receipt is `provider-accepted-status-v1.txt` SHA256 `1d7b5a2fb8a41404909dd2e5de154a2dedcc0ab5c52e51ec4331a7e6ca3adea0`. No second `push` occurred. The first CLI error is retained separately in `provider-first-push-error-v1.txt` SHA256 `4f8933891f78c81142502b48ed705fa4425c7650100324317e03bd2c5b8b045c`.

Provider notebook URL: [https://www.kaggle.com/code/abdulazizshehri/hala-h47-paired-prompt-dev-eight](https://www.kaggle.com/code/abdulazizshehri/hala-h47-paired-prompt-dev-eight).

**QUEUED does not mean inference happened**, nor that real model accuracy improved. Wait for original terminal v1, preserve all 16 raw trials including errors, then independent paired scoring; do not resubmit just because Kaggle client returned a decoding error.
