# H399 — private original-synthetic Qwen3-VL-2B LoRA research

**Separate non-release model-adaptation pilot** on the owner's free Kaggle NVIDIA T4. This is not Hala flagship training and cannot qualify any original application screenshot, OOD benchmark or weights for release. H199's independent rights-admission list remains intentionally EMPTY.

Upstream public open-weight base: `Qwen/Qwen3-VL-2B-Instruct` at immutable Hugging Face revision `89644892e4d85e24eaac8bacfd4f463576704203`, verified against the official model repo metadata as Apache-2.0. All 24 TRAIN, 6 DEV and 6 author-exposed alternate-style test images are generated entirely by `h399/kaggle/h399_original_qwen3vl_synthetic_lora.py` using first-party colored rectangles. Source identities are checked for disjointness. 12 English and 12 Arabic train instructions; no third-party rendered GUI screens, real user data, copied typography, or private Kaggle datasets.

## Verified owner Kaggle outcome (2026-11-03)

**COMPLETE, real 2.127B parameter upstream vision-language base** `Qwen/Qwen3-VL-2B-Instruct` with private rank-4 language-only LoRA. Kaggle returned **16/16 actual optimizer steps**, a saved PRIVATE 3,227,648-byte `adapter_model.safetensors` containing **112 LoRA tensors**, including 56/56 inspected nonzero `lora_B` matrices. The downloaded local inspection copy was deleted; trained adapter remains in private Kaggle output only. The model revision and Apache-2.0 metadata were independently checked via official Hugging Face API.

Measured completion loss (lower is better), **not end-to-end click accuracy**:
- Original-only DEV6: base **0.804674** → adapted **0.398591**.
- Author-exposed alternate-style synthetic6: base **1.132086** → adapted **0.391298**.
- TRAIN24 bilingual invented grid cases; developer observed every source family. Total Kaggle recorded run time 95.265 seconds including package/model start, not a general throughput claim.

The exact Kaggle owner receipt JSON plus SHA256 hash are included under `h399/verified_receipt`; `python -B h399/validate_receipt.py` validates it without running a GPU. Upstream weights were downloaded via Kaggle Internet access; **no original GUI images or private datasets were supplied, attached or downloaded**. The research adapter is not a Hala product model. Kaggle logs contained an upstream PyTorch gradient-checkpoint warning about some input gradients; independent inspection of the saved adapter confirmed **56 nonzero LoRA-B matrices**, but further full visual grounding tests remain necessary.

These numbers **do not** demonstrate real GUI task completion, unseen external OOD generalization, improved Hala flagship weights, legal training authorization for original screenshots or publishable benchmark gains. The formerly quoted experimental status 'may fail' is superseded by the SHA-verified completed receipt.

The kernel is private, with `dataset_sources: []` and `model_sources: []`, but **Internet is enabled** solely to install Python dependencies and download the public Apache-2.0 model revision; no data is sent to an inference API. Only an official rights review can authorize future training on real browser screenshot sources. External legal reviewer issue #384 and blinded independent bilingual OOD custodian issue #385 remain open; Hala release stays NO-GO.

Kaggle: https://www.kaggle.com/code/abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora

Instructions:

```powershell
kaggle kernels push -p h399/kaggle --accelerator NvidiaTeslaT4 --timeout 2400
kaggle kernels status abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora
kaggle kernels logs abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora
```
