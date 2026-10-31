# H266 — repair first real private Kaggle GUI runtime blocker

H265's first owner-authenticated private Kaggle `NvidiaTeslaT4` execution reached the model/browser runtime but **failed before native GUI actuation**. The browser launch traceback identified the missing Linux system library `libatk-1.0.so.0`; `playwright install chromium` had downloaded browser binaries without the required OS libraries.

H266 changes the private payload builder to run the documented `python -m playwright install --with-deps chromium` instead. This installs Chromium's required system packages and browser binaries inside the **ephemeral Kaggle Linux environment**, never on the owner Windows machine. The existing fail-closed T4 guard, pinned public H264 source and first-party H262 app, private kernel metadata, no-source-media upload rule, and bounded single pretrained generation remain unchanged.

Evidence: private Kaggle status `ERROR` on first attempt; error class `playwright._impl._errors.TargetClosedError`, missing `libatk-1.0.so.0`. No screenshot, gold, raw prediction, source receipt, model weights or credentials are included in this public PR. Reference: [official Playwright Python browser dependency instructions](https://playwright.dev/python/docs/browsers#install-system-dependencies).

**Qualification boundary:** A local payload-generation test checks that `--with-deps` is present, but it does not prove that Kaggle's apt repository or runtime will permit dependency installation. A fresh private free-T4 rerun is necessary. The first attempt yielded **zero measured GUI task success and no validated model gain**. H173 remains SEALED; H199 admitted TRAIN remains empty. H266 is a runtime repair, not a trained-model improvement.

## Independent verification — Kaggle private version 2, 2026-10-31

The H266 fixed Chromium system dependencies and the owner-private Kaggle
kernel reached **COMPLETED** (not merely successful upload). The
create-only **aggregate-only** JSON was retrieved with an exact-file
filter and privately validated; no screenshot, raw model response,
reference target or per-case gold was downloaded or published.

Measured facts from that one first-party research trial:

- Actual free Kaggle NVIDIA **T4 x2** hardware verified by CUDA.
- Pinned pretrained **Qwen/Qwen3-VL-2B-Instruct**, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- **1 actual completed model.generate call** with real screenshot input.
- `output_class = INVALID_ACTION`; native action not dispatched.
- First-item inventory count **did not increase**, so the task
  succeeded **0/1** under the strict native action contract.
- **0 optimizer steps**, no Hala adapter loaded, no TRAIN/DEV/OOD
  admission, no independent blinded or historical H173 OOD touched.

This is first empirical proof in this run that real pretrained Qwen
inference can execute in the complete ephemeral Kaggle T4 + Chromium
source-native environment; it is also an explicit negative baseline
for strict JSON/native GUI actuation. It is **not** proof of a better
trained Hala model. We do not silently strip model reasoning, patch
invalid JSON, provide an oracle-derived fallback, or count failed
dispatch as task success. A follow-up may investigate the **format
failure** safely with private, aggregate-only diagnostics or an
independent research TRAIN pilot; do not reuse consumed H173 OOD.

Hosted H266 CPU CI run #41192741444: **COMPLETED SUCCESS** at initial
head `0586426573836fc2659f1061155442779dab826c`; Python
3.11.14 and 3.12.14 unittest jobs both passed. CPU CI does not
verify the Kaggle model; the private aggregate is a distinct source.
