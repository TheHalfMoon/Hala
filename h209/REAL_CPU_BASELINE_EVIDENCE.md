# H209 — first public pre-trained VLM on self-generated pixels (CPU baseline)

**Draft-only research** above H208 Draft #229 at exact commit
`ba8adfbe0fac4ce95ff52d3799076681d815cce0`.
This is **not Hala training, a new Hala checkpoint, a GUI action model
accuracy evaluation, TRAIN/DEV admission or official held-out OOD.**

## Why this model / hardware

The owner's Windows 11 machine has about 15.7 GiB system RAM, Intel UHD
Graphics and no NVIDIA CUDA GPU, with limited free disk. The existing
`Qwen/Qwen3-VL-2B-Instruct` Hugging Face cache contains tokenizer,
processor and config files but **no model weight shard**. Do not claim
H208 ran Qwen3-VL for real.

Instead, H209 uses an *independent, explicitly baseline-only* model,
`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, in a disposable, guarded
CPU research environment. The Hugging Face model metadata was checked
online on the owner's Windows host, including immutable official
revision `9b299836fe84e0aabd39e1e07651e569e5dfe87c`,
Apache-2.0 license, and approximately **1,025,998,224 bytes** of
safetensors. The runtime is Python 3.11.15, PyTorch CPU 2.7.1,
torchvision CPU 0.22.1, Transformers 4.57.3, Pillow 12.3.0,
Accelerate 1.11.0 and num2words 0.5.14.

No user data, historical screenshots, source gold, browser/Dom,
old H173 OOD or reserved Gantt are used. All HF model/cache files are
restricted to an isolated temporary folder on **D** with a hard disk
free-space guard; the code refuses another path or cache location.
The script must be explicitly invoked on an owner device. CI never
downloads weights, accesses private roots or performs model inference.

## Real model protocol vs real learning

`h209/public_cpu_visual_baseline.py` constructs two **new in-memory**
960×720 images, each with a large red/blue rectangular region and
otherwise plain white pixels. It sends each PIL image and an
English color instruction to the pinned public pre-trained model via
`AutoProcessor.apply_chat_template`, verifies `pixel_values`
are present and nonzero, invokes `model.generate` on CPU with
deterministic decoding, and validates the prompt prefix before decoding
the continuation only. It prints only aggregate red/blue correctness,
number of model calls, decoded visual tensor dimensions and new tokens:
never any owner screenshot, raw model response or private source SHA.

Any number of correctly named colors is a **tiny known synthetic
diagnostic**, not GUI localization accuracy, not independence of the
training corpus and not OOD generalization. The downloaded public model
is **not Hala's trained model**, so the number cannot be reported as
Hala accuracy or used to approve H199 training.

## Owner observed gates

- Initial source-only synthetic tests: **7/7 normal and 7/7 Python -O**
  passed, including pure raster generation, pairwise color divergence,
  explicit fixed public model pin, missing visual tensor and
  generated-token continuation shape.
- Official Hugging Face metadata pin and Apache-2.0 license checked on
  owner Windows; printed only model ID/commit/license/weight bytes.
- Initial actual VLM attempt stopped at `SmolVLMProcessor` import
  because PyTorch CPU did not install `torchvision` transitively.
  It made **zero** model calls. Compatible `torchvision==0.22.1+cpu`
  was then installed into the same disposable venv.
- **Second actual Windows CPU run, no mock/model fixture: SUCCESSFULLY
  LOADED PINNED PUBLIC MODEL AND GENERATED TWO CONTINUATIONS.**
  Exact source output: `H209_REAL_PUBLIC_CPU_WEIGHTS_AND_PROCESSOR_LOADED`
  at commit `9b299836fe84e0aabd39e1e07651e569e5dfe87c`.
  The two real generated-color diagnostics were:
  `target=RED recognized=True correct=True generated_tokens=2
  visual_elements=10223616` and
  `target=BLUE recognized=True correct=False generated_tokens=2
  visual_elements=10223616`. **1/2 correctly answered, not 2/2**.
  Completion receipt:
  `H209_PUBLIC_PRETRAINED_REAL_INFERENCE_FINISHED model_calls=2
  self_generated_color_answers_correct=1 new_hala_training_steps=0
  official_ood_calls=0 private_media_opened=0 no_gui_action_claim=true
  elapsed_seconds=175.6`.
- This baseline is **poor and non-conclusive as a vision benchmark**:
  only two hand-built color panels, no blind tasks or original GUI
  action reward. It cannot count as a model-generalization score,
  source-disjoint held-out test or Hala progress in accuracy. Crucially,
  do **not** hide its observed incorrect BLUE answer.
- Hosted 3.11/3.12 GitHub Actions only run the seven tiny pure
  synthetic contract tests (normal and `-O`), no model download,
  PyTorch/transformers/weights or private data. Its exact-head CI result
  must be independently checked; Windows CPU inference is separate.

## Remaining Hala work (not solved by public baseline)

For the actual Hala vision model, separately pin and qualify
Qwen3-VL-2B processor/model revision, load appropriate weights with
genuine free capacity, certify screenshot-to-token parity, and obtain
measured genuine native GUI reward on independent original applications.
The 20 quarantined original GUI owner cases remain unadmitted,
old pre-H173 decoded RGBA and fork/source completeness remain
unproven, and H199 official TRAIN allowlist remains EMPTY. Do not
open consumed H173 OOD, fabricate old gold, run paid compute,
merge to main or publish a model.
