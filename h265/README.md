# H265 — real, private, free Kaggle T4 Qwen3-VL first-party probe

**No model or weights are published. No original independent TRAIN/DEV/OOD,
H173 consumed scored OOD, H212 barred examples, owner-private media or
reference gold is used or uploaded.**

This is a directly runnable, controlled **pretrained BASE** model inference
experiment, intentionally separate from proving a fine-tuned Hala model
improves in the wild. The exact first-party app H262 source and H264
screenshot-locked browser are used. The remote notebook remains PRIVATE.

## Why this step

Earlier H15 established real, bounded LoRA *training mechanism* success
(two optimizer steps and resume equality) but not any change in GUI skill.
H263 dispatched native actions from a synthetic source policy; H264
revalidated the exact screenshot before actuation. H265 attempts to
replace the synthetic policy with **actual Qwen3-VL-2B-Instruct pretrained
model.generate** on owner-authorized free Kaggle T4 hardware.

The experiment tests a first-party Stockroom GUI task: click the +1 button
of the first item. The pinned original self-authored app is downloaded as
the exact public H264 commit, checked clean, and the SHA256 of the
unchanged H262 HTML source checked before model/browsing. No third-party
media source, old 32-case OOD, old source screenshots, or private reference
items are mounted. Playwright generates one real fresh source screenshot
IN RAM on Kaggle; H208 feeds pixels through the full HF multimodal
processor and genuine pretrained model. H264 validates strict one-step
JSON, source/URL/viewport, fresh screenshot identity, and executes one
H263 browser action if valid. Only AFTER model output does a trusted,
independent test oracle read the first inventory quantity to determine
whether it incremented by exactly one. This oracle is NEVER passed to the
model. If generation/JSON/action fails, record failure; no oracle fallback.

A wrapper counts a **successfully completed model.generate** call. CPU
tests of that wrapper are NOT a real model run; only an independently
verified remote Kaggle result can claim neural inference.

## Exact safety / compute constraints

- Private Kaggle kernel only, explicit noncommercial opt-in.
- **NvidiaTeslaT4** free T4 accelerator, verified with real CUDA names,
  not paid GPUs or CPU pretending to be GPU.
- Model Qwen/Qwen3-VL-2B-Instruct pinned revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Transformers **5.16.1** pinned, Playwright **1.58.0**, Kaggle's
  existing CUDA PyTorch retained. No local Windows/Mac model download.
- Single source GUI initial scene, single capped 160-token decoding.
- Images remain in remote ephemeral memory only. No raw predictions,
  private screenshots, reference coordinates, user tasks, gold or
  credentials in logs, reports, GitHub or Kaggle files.
- Aggregate result contains generation count, coarse action kind,
  whether native input occurred and a first-party stock delta success
  flag, plus explicit false flags for independent OOD, official data
  admission, training, model adaptation and release qualification.
- If hardware, package, model revision, source, or network conditions
  block the run, retain the **real failure**. Do not substitute a
  fake GPU test or fabricated model success.

## Reproduction

On the authorized owner Windows with Kaggle CLI already connected,
from the Hala checkout, create a new scratch directory outside Git:

    python -m h265.build_private_kaggle_payload --source h265/free_kaggle_pretrained_probe.py --destination D:\Hala-H265-Kernel-Upload
    kaggle kernels push -p D:\Hala-H265-Kernel-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h265-private-qwen3vl-firstparty-t4

The scratch upload contains only ONE generated Python script and
kernel-metadata.json. Kaggle metadata has **is_private=true**, GPU T4,
and zero dataset or model sources; model public weights are fetched
only by the ephemeral remote kernel. Delete scratch locally after
successful verified source upload; keep all reproducible logic in Git.

The H265 public tests verify safety contracts and payload construction,
not any pretrained output or model gain. We can claim actual model
inference only if the Kaggle remote run returns an independently checked
`real_pretrained_model_generations=1` with genuine CUDA qualification.
Even if first-party task success is true, this is **not** a source-disjoint
blind OOD, a trained Hala adapter, or an independent model gain.
