# H88 — Reproducible private Kaggle research training launcher

H87's actual Kaggle script previously existed only in an inaccessible Windows workspace, not in GitHub, and two genuine provider failures were fixed only in that local script: **`torch.cuda.get_device_count()` is not a PyTorch API** (use `torch.cuda.device_count()`), and Kaggle's `torchao==0.10.0` optional package caused `peft==0.20.0` to throw after Qwen loaded. H88 preserves these fixes as a reproducible, source-pinned research launcher.

## Invocation on an authorized research workstation

From the H88 GitHub checkout, with its own authenticated Kaggle CLI:

```bash
kaggle kernels push -p h88 --accelerator NvidiaTeslaT4 --timeout 2400
```

The included `kernel-metadata.json` keeps the notebook **private**, attaches **only** the existing owner-private H71 TRAIN dataset, selects T4, and enables internet solely so immutable public model weights/source can be fetched. This command is an explicit submission; the presence of the files in GitHub **does not** submit or schedule Kaggle itself.

The script uses the already-reviewed H87 source SHA `5a8396881b655736277b25255d7101f6c602213e`, not a moving branch. It checks actual CUDA, removes the optional incompatible `torchao`, verifies exact Torch/Transformers/PEFT/Pillow pins and Git source, and then calls the original H87 trainer (one true Qwen3-VL-2B LoRA optimizer step over exact, verified H71 original TRAIN). It writes a bounded `hala-h88-research-attempt.json` metadata receipt showing stage and failure class without including raw prompts, secrets or training data.

**Do not interpret an accepted job as a trained model.** Only success after H87 independently loads the immutable adapter+optimizer checkpoint may be marked `REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED`. Provider-private independent readback and bilingual held-out GUI metrics are still separate unsatisfied requirements. Leave outputs private and do not merge/release.

## Evidence

Seven H88 local isolated CPU tests passed using a standalone copy of the committed launcher and metadata: safety gates, proper version pins, correct `torchao` removal order, private Kaggle metadata, safe fail receipts and missing-accelerator refusal. GitHub CI tests on 3.11 and 3.12 should be checked after PR creation. **The H88 launcher itself has NOT executed on Kaggle in this change**: the Windows Desktop Commander connection is over its monthly quota, TinyFish currently has no Kaggle login, and the available Opera connector is offline. We have not verified final status of the previously submitted H87 Kaggle v4 job.

Note: H88 is for personal internal noncommercial research. Its outputs are not approved for commercial training or deployment.
