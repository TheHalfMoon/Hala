# H216 — genuine offline Qwen3-VL processor-only multimodal SFT parity

**Strict Draft stack** based on H215 PR #236 exact verified
`d669e0e5c9ca9a6114815a7480f3a0ce7bd9d596`.
No main merge, release, cloud inference, GPU optimizer, model weights,
private media, historical gold or consumed H173 OOD.

## Why this experiment matters

H212 produced 96 entirely new and research-only 960x720 original GUI
Pillow images (48 English tasks, 48 Arabic tasks; 4 layout families).
H215 protected their frozen provenance on Windows/Linux (4 hosted
specialized matrix jobs plus 2 general CPU checks green on PR #236).
Previous H199 `assistant_completion_mask` tests used fake token IDs,
while the H208 preloaded policy tested only fake HF processors.
**Before touching any model training**, this H216 tests whether the
real, locally cached Qwen3-VL processor tokenizes the *actual* H212
image+prompt into the exact prefix of the user+assistant SFT sample.

## Pinned real processor, deliberately zero model calls

- **Owner Windows 11 only**. Existing local HF hub cache
  `Qwen/Qwen3-VL-2B-Instruct` snapshot revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
  Cache contains only processor, config, tokenizer and chat template:
  **no 2B pretrained model weights are used/downloaded**.
- Disposable isolated `D:\Hala-H216-QA` CPU Python 3.11 environment,
  PyTorch **2.7.1+cpu**, torchvision **0.22.1+cpu**,
  Transformers **4.57.3**, Pillow **12.3.0**,
  Accelerate **1.11.0**, num2words **0.5.14**.
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `local_files_only=True`, `trust_remote_code=False`.
- Source `h216/real_qwen_processor_alignment.py` imports only
  `AutoProcessor`; neither `AutoModel` nor `from_pretrained` on
  any **model weight** nor `generate`, `loss.backward`, trainer,
  optimizer or model save is reachable.
- The only image inputs are `research_multimodal_messages(index)`
  from H212, which generate fresh PIL screenshots in-memory. H215's
  frozen source verifier first checks **96/96** original Windows PNG
  and decoded RGBA digests (on owner Windows).
- For a fixed subset of four independent EN/AR examples and an
  optional full source census of 96 cases, the real Qwen3-VL processor
  tokenizes user/image prompt with `add_generation_prompt=True` and
  user/image+strict native click answer with
  `add_generation_prompt=False`.
  Both use `tokenize=True`, `return_dict=True`,
  `return_tensors="pt"`, and the **actual PIL image**.
- Requires identical nonempty actual `pixel_values` and
  `image_grid_thw` tensors in the prompt and full sequence,
  exactly one token row, **strictly identical full-prefix token
  IDs**, a nonempty assistant completion, all question/vision tokens
  masked as `-100`, all assistant tokens unmasked, and
  continuation-only `batch_decode(skip_special_tokens=True)`
  exactly parsed as H199 native one-click JSON.
  Rejected data MUST NOT be silently truncated/aligned heuristically.
- Never prints raw token IDs, source click coordinates, PNG contents,
  screen images or private metadata. Evidence is only aggregate counts.

## Actual measured first 4 owner Windows samples

The frozen source and genuine `Qwen3VLProcessor` **loaded offline**.
All four independently generated PNG+PIL examples were processed with
real nonzero vision tensors:

| H212 source index | Language | Prompt token count | Full token count | Assistant-only tokens | Vision tensor elements | Prefix and click JSON |
|---|---|---:|---:|---:|---:|---|
| 0 | English | 839 | 859 | 20 | 4,055,040 | PASS |
| 1 | Arabic | 849 | 869 | 20 | 4,055,040 | PASS |
| 64 | English | 839 | 859 | 20 | 4,055,040 | PASS |
| 65 | Arabic | 849 | 869 | 20 | 4,055,040 | PASS |

All four returned exact prefix equality, identical pixel values/grid,
assistant-only loss masks, and strict native click JSON roundtrip.
**This is real model-input preprocessing, not model inference**.

## Expanded full-96 source audit

**Completed on owner Windows 11 using the REAL cached
Qwen3VLProcessor** with offline mode and the actual H212 PIL pixels:
`python -m h216.real_qwen_processor_alignment
--local-processor-snapshot <exact-local-cached-revision>
--all-research` (the path was supplied only to the local process;
it is not committed).

- Every one of the **96/96** in-memory research candidate images from
  toolbar 32, sidebar 32, dialog 16 and footer 16 passed actual
  multimodal processor SFT source checks.
- Every case had exact **prompt-prefix ID parity** between the
  generation-mode question and training-mode question+assistant.
- Every case had actual identical nonempty `pixel_values` and
  `image_grid_thw` tensor values, no task row or vision-mapping loss,
  all prompt/vision IDs masked to `-100`, an unmasked native-click
  assistant continuation, and a strict decoded native JSON click.
- At each actual 16-case progress checkpoint, the report showed
  **20 supervised assistant tokens** and **4,055,040 genuine
  vision tensor elements** for the reported example.
- Final observed receipt:
  `H216_QWEN3VL_H212_REAL_PROCESSOR_ALIGNMENT_PASS
  synthetic_cases=96 source_rgba_verified=96 model_calls=0
  optimizer_steps=0 official_train=0 official_dev=0 official_ood=0`.
- Elapsed owner process runtime: **33.65 seconds** (processor and
  tokenizer only, no model weights). No owner private screenshots,
  gold, consumed OOD, model generation or training.
- The 96 candidates are intentionally **NOT** an independent
  validation set or officially admitted TRAIN corpus. This is
  *only actual source/processor tokenization parity*, not model
  performance, trainability, memory feasibility of 2B weights,
  accuracy or any downstream reward.

## Independent source-only CI and scientific limits

- `tests/test_h216_real_qwen_processor_alignment.py` uses
  **explicit fakes** only to test input-loss-mask safety and fail-closed
  rejection for prompt mutation, image tensor or grid mutation,
  malformed JSON, extra rows, invalid sample indices, absent
  offline assets, and forbidden training admission. Such tests DO
  NOT establish real Qwen behavior, which is separately observed
  in the Windows-only owner CPU run above.
- The H216 hosted Linux CPU job is Pillow-only and
  `HF_HUB_OFFLINE=1` plus `TRANSFORMERS_OFFLINE=1`. It never
  downloads model/processor, loads weights, or accesses owner data.
- **No training, no checkpoint, no measured GUI action accuracy**.
  Full pretrained Qwen3-VL weights are not qualified/present.
  No official TRAIN/DEV/OOD data is admitted; H199
  `ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()` remains EMPTY.
  20 owner original source prospects still quarantined, and
  full old source/fork + decoded RGBA exclusions missing.
  H173 consumed OOD and reserved Gantt OOD must stay sealed.
- Model performance remains unmeasured on H212; the previous
  distinct public pretrained 256M model yielded **0/2 strict
  actionable GUI JSON** in H211, not Hala accuracy.
