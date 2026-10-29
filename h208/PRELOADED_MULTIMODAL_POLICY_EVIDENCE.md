# H208 — offline preloaded Qwen3-VL-style multimodal policy adapter

**Research-only engineering.** Stacked exactly on H207 Draft #228 at
`38cb41ae5fdb6be58067ff788e9b2edf0c4f5f08`.
No checkpoint loaded, no actual learned inference, no optimizer steps,
no GPU, new model benchmark or TRAIN/DEV/OOD release.

## Practical new model integration

H207 accepts `predict(png_bytes, screenshot_task_prompt) -> action_json`
but did not implement a real multimodal processor + model call.
`h208/preloaded_vlm_policy.py` implements this callable contract for
an already-loaded Hugging Face Qwen3-VL-style `AutoProcessor` and
`Qwen3VLForConditionalGeneration` model.

- Validates exact 960×720 bounded PNG and **actually decodes full
  original RGB raster** using Pillow in memory. It is not satisfied
  by a forged PNG header.
- Passes a chat user message containing the **actual decoded PIL image
  and same H199 screenshot-only prompt** to
  `processor.apply_chat_template(... tokenize=True,
  add_generation_prompt=True, return_dict=True, return_tensors="pt")`.
- Requires both `input_ids` and nonempty/nonnull `pixel_values`;
  no text-only fallback. Batch is transferred to the preloaded
  model's device.
- Invokes `model.eval()`, `torch.inference_mode()`,
  `model.generate(do_sample=False, num_beams=1,
  max_new_tokens=160)`. Output is batch-size-one only.
- Verifies generated tokens start with exact input prompt token IDs
  and extracts **only new continuation tokens**, as in the
  documented Hugging Face Qwen3-VL model usage. A changed prompt
  prefix or empty/overlong continuation is rejected.
- Decodes the continuation and uses H199/H200's bounded strict native
  action parser. Explanations, unexpected fields, unsupported keys,
  duplicate keys and nonstandard NaN/Infinity never reach native input.
- Model output/errors are redacted; raw screenshot and output are not
  written to disk, uploaded or used as gold. H207 still guards browser
  URL, viewport, fresh screenshot SHA and native input.

Official documentation and model card used:
- https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

The target public base is `Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0).
No immutable model revision, local model weights, model-processor
identity, training data release or model output accuracy is attested.
Unlike a real loader, this class deliberately does NOT call
`from_pretrained()`, cannot silently download massive weights,
and cannot start Kaggle/paid inference.

## Actual owner Windows test evidence

Using isolated Python 3.11.15 + Pillow only, deterministic fake
`torch.inference_mode` context and fully generated non-private
960×720 PNGs, both normal and `-O`:

- H203 prior screenshot freshness: **11/11** each.
- H207 prior screenshot-only native adapter: **14/14** each.
- H208 new processor/model-shaped synthetic protocol: **13/13** each.
- **38/38 per mode**, **76 passes** over two modes.
  Includes real Pillow raster change propagation, image tensor
  mandatory, prompt prefix equality, continuation-only decoding,
  bad model outputs, missing token budget, malformed image,
  abstention and H207 synthetic end-to-end dispatch.
- No real PyTorch, HF processor, genuine Qwen weights, model inference,
  or GPU ran during those tests. No training or native accuracy claim.
- CI: dedicated free hosted Linux Python 3.11/3.12 synthetic only;
  **its pass must be observed at exact commit separately.**

## Remaining prerequisites for a genuine Hala model evaluation

1. Independently pin the model and processor SHA/revision. Verify
   model adapter license and exact inference backend behavior.
2. Connect legally usable existing weights or genuinely free,
   user-controlled Kaggle/Colab GPU session; no unapproved paid jobs,
   downloads to space-constrained Windows, secret leaks or private data
   uploads. Explicitly record CUDA/T4 memory and processor token parity.
3. Run H208 with a real vision model on newly created *non-held-out
   public/synthetic GUI-only research tasks* first; record all model
   action outputs and native original rewards without overclaiming.
4. Before any official fine-tune/new model accuracy: prove complete
   historical original source/fork and old RGBA exclusions, officially
   admit role-disjoint TRAIN/DEV, seal labels and select fresh independent
   OOD. H199 official TRAIN allowlist is EMPTY and the existing 20
   owner source prospects remain UNADMITTED. Never reopen consumed
   H173 OOD or the reserved Gantt tasks.

An injected `model` could ignore pixels even when it receives a
visual tensor; this module only verifies the **protocol path**, not
learned visual grounding or model identity. It is not a sandbox
for adversarial model code and its browser action is not atomic.
