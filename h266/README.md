# H266 — repair first real private Kaggle GUI runtime blocker

H265's first owner-authenticated private Kaggle `NvidiaTeslaT4` execution reached the model/browser runtime but **failed before native GUI actuation**. The browser launch traceback identified the missing Linux system library `libatk-1.0.so.0`; `playwright install chromium` had downloaded browser binaries without the required OS libraries.

H266 changes the private payload builder to run the documented `python -m playwright install --with-deps chromium` instead. This installs Chromium's required system packages and browser binaries inside the **ephemeral Kaggle Linux environment**, never on the owner Windows machine. The existing fail-closed T4 guard, pinned public H264 source and first-party H262 app, private kernel metadata, no-source-media upload rule, and bounded single pretrained generation remain unchanged.

Evidence: private Kaggle status `ERROR` on first attempt; error class `playwright._impl._errors.TargetClosedError`, missing `libatk-1.0.so.0`. No screenshot, gold, raw prediction, source receipt, model weights or credentials are included in this public PR. Reference: [official Playwright Python browser dependency instructions](https://playwright.dev/python/docs/browsers#install-system-dependencies).

**Qualification boundary:** A local payload-generation test checks that `--with-deps` is present, but it does not prove that Kaggle's apt repository or runtime will permit dependency installation. A fresh private free-T4 rerun is necessary. The first attempt yielded **zero measured GUI task success and no validated model gain**. H173 remains SEALED; H199 admitted TRAIN remains empty. H266 is a runtime repair, not a trained-model improvement.
