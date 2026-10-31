# H270 — coordinate-weighted visual grounding pilot (private free T4)

**Experimental first-party research; no model release, official TRAIN/DEV/OOD admission or generalizable quality claim.**

## Motivation and preregistered design

The genuine H269 owner-private free T4 trial completed two real rank-4 FP16 LoRA optimizer steps, with 112 adapter tensors modified in each step, but paired Arabic original Stockroom GUI task failed before and after (0/1 vs 0/1). Two steps and two simple frames yielded no measured visual target improvement.

H270 changes the supervision rather than merely formatting:

- EIGHT truly rendered original Stockroom screenshots for eight author-owned TRAIN labels. Scenes: default English, sorted English, four English searches, default Arabic, sorted Arabic. Source-owned DOM target geometry is extracted ONLY on TRAIN; supervised native JSON contains one click on first visible +1 button. Model callback receives screenshot pixels plus task only, NEVER DOM/gold/geometry/fallback.
- Pinned pretrained Qwen/Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203; pinned original HTML SHA256 3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a; original Git H269 SHA 16ef17ac957b345dbc30727370f3a98411db4c1f.
- Hala training/scripts/train_adapter.prepare_inputs builds completion-only masks. Existing h61.supervision.weighted_lm_loss gives coordinate digit tokens EIGHT times the weight of structural JSON tokens. EIGHT true rank4 language LoRA FP16 steps, LR 5e-5, with finite positive loss, coordinate masks, nonzero gradients and actual adapter changes required on each step.
- BASE and ADAPTED evaluated separately on two different Arabic RTL search+sort scenarios, different from ALL TRAIN PNG bytes, with pixel-identical screenshots across each paired evaluation. H208 real visual model, H268 strict token-level JSON grammar, H264 screenshot stale lock and H263 real native browser action, with trusted inventory state checked ONLY after infer/actuation. No oracle answer to policy.
- Report ONLY aggregate evidence: actual model/optimizer completion; coarse native action dispatched and task-state outcome; explicit FALSE flags for generalizable improvement, independent OOD, official admitted data, raw images/gold/weights publication.

The H262 original Stockroom and task were already exposed in prior research. These are ONLY byte-disjoint first-party scenes, NOT a new blind independent source-family OOD benchmark. Even a positive two-case delta cannot establish generalizable skill. A negative outcome remains negative.

## Privacy, repeatability and zero cost

Code-only PRIVATE Kaggle NVIDIA Tesla T4 free noncommercial research, zero dataset/model/weight/media attachments. Fixed Transformers5.16.1, PEFT0.20.0, pinned MIT LMFE core public Git commit 817f944fcc8851917c40300a47f62d1defc3ffc3. Keep Kaggle original CUDA PyTorch unchanged. The narrow H269 optional torchao==0.10.0 incompatibility fix runs only in ephemeral Kaggle.

Model weights, original source, TRAIN PNGs, click targets, optimizer and adapter stay ONLY in remote ephemeral /tmp/hala-h270-firstparty and are deleted at exit. The ONLY allowed output is small private aggregate JSON named hala-h270-private-weighted-firstparty-aggregate.json under /kaggle/working. NEVER use unfiltered Kaggle output downloads.

Owner Windows real installed Edge preflight: 10/10 valid original browser scenes, 10/10 distinct PNG byte contents, no PNG persisted and no EVAL target coordinates extracted. Local related CPU tests: H270/H269 focused 9 PASS; broad H199/H201/H207/H208/H249/H262/H263/H264/H265/H267/H268/H269/H270 133 PASS, 3019 unrelated deselected; compileall clean. These are NOT model success evidence.

Reproduce on the already authorized owner Windows Kaggle CLI by building the private 2-file package from h270/build_private_payload.py; destination D:/Hala-H270-Private-Upload (fresh only). Submit with 'kaggle kernels push -p D:/Hala-H270-Private-Upload --accelerator NvidiaTeslaT4'; monitor private owner slug abdulazizshehri/hala-h270-private-coordinate-weighted-lora-t4. Fetch only the exact filtered small JSON after COMPLETE. Delete temporary upload scratch.

## Permanent scientific barriers

H199 ADMITTED_TRAIN_MANIFEST_SHAS EMPTY. H173 old 32 scored cases permanently CONSUMED and SEALED. H212 scored procedural rows barred. H250/H251 other original apps private and not admitted due missing transitive assets licenses/fork/npm provenance and PNG+decoded RGBA visual exclusion. A real Hala release needs legally qualified multi-original-family TRAIN/DEV, newly untouched blind independent OOD and reproducibly measured generalizable native task-state improvement.


## Real private Kaggle v1 failure and exact FP16 repair

The owner-private H270 Kaggle FREE T4 v1 was genuinely RUNNING and then
ended ERROR before completing the eight-step qualification. The
sanitized Python traceback showed H270's explicit fail-closed
ValueError "nonfinite/zero coordinate gradient"; no optimizer success
or base/adapted score is claimed for v1.

Source audit found the difference from the pre-existing successfully
qualified Hala trainer: training/scripts/train_adapter.py sets the
FP16 GradScaler initial scale to **1.0**; H270 v1 had inherited
PyTorch's high default scale, which can overflow in coordinate-weighted
FP16 backprop. The isolated H270 private kernel now uses
GradScaler("cuda", init_scale=1.0, enabled=True), exactly the
previous Hala trainer setting; the regression suite compares both
code paths. No change to model, GPU hardware, target coords,
loss weights, allowed-token grammar, original source, or dataset
admission. Real v2 success must be reverified independently; this
source repair does NOT itself demonstrate a valid LoRA step.
