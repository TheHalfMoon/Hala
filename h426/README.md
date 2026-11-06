# H426 — Original H425/H421 geometric-error attribution and loss prototype

Status: **PRIVATE CPU research engineering only**, NOT a newly-trained H426 model, production-ready agent, legally admitted training set, or independent OOD result.

## Authentic model evidence

Reference authentic owner **private** provider receipts (never commit the originals):
- H425 true trained adapter SHA: `bc123aa521d58e35a205805759e38c8a6a4af4d62fc01ac3e118cc7aab7f4957`
- H425 model source training receipt SHA: `14c48fcfe567c8c30465cc08ce323aba6841d4bb67bdfc30cc8750a3940a1057`
- H421 original two-prompt model evaluation receipt SHA: `eaef49bf17e0de87279397607ab7269be50235acdbfc8e0038c2f4c70f7af9f5`
- H425 original source-heldout 240 cases 94/240 H417 → 114/240 H425; H417 old144 76→81, fresh96 18→33, original Arabic24 24→24.
- H421 *author-heldout English* 192 new source JPEG×2 instructions (384 actions), H417 individually 86→H425 172 correct, both correct 11→45 **of 192 pairs**. This is research DEV, not independently sealed external OOD.

**Independent original source/model verification precedes this audit**: `h425/validate_h421_pair_receipt.py` replays the original 384 source-target correctness decisions from original source JSONL/receipt + GPU receipt and verifies real H425 adapter/receipt SHA. H426 recomputes exact model-vs-source scoring again, then produces **aggregates only**, never raw prompts, individual IDs, gold answers, or screenshots.

## H425 error attribution — exact original scorer ±50 normalized/1000

Every source-domain group contains 128 true original evaluated instructions. Both models' 384 predictions were schema-valid, so the actionable bottleneck is source-grounded geometry rather than JSON grammar.

| Original domain | H417 correct | H425 correct | H425 end-only error | H425 start-only error | H425 both-point error |
|---|---:|---:|---:|---:|---:|
| slider | 52 | 94 | 32 | 1 | 1 |
| slide_resize | 10 | 35 | 63 | 1 | 29 |
| text_highlight | 24 | 43 | 17 | 8 | 60 |
| **Total** | **86** | **172** | **112** | **10** | **90** |

Original per-image correct action transition (H417 count→H425 count, out of 192 pairs):
`0→0:63, 0→1:38, 0→2:16, 1→0:2, 1→1:41, 1→2:21, 2→1:3, 2→2:8`. This preserves regressions, not only wins. Diagnostic result includes no private case-level data.

## CPU-engineered H426 hypothesis (NOT trained or shown to improve)

`h426/target_aware_loss.py` supplies a pure per-supervised-token numeric field-specific weighting proposal:
- baseline JSON grammar tokens 1.0; current H425 coordinate token weight 8.0
- **slide_resize** endpoint numeric target tokens weight 16.0, start remains 8.0, to address 63/128 end-only failure cases
- **text_highlight** both start/end numeric tokens weight 12.0, to address 60/128 both-point errors
- **slider** both coordinate fields remain 8.0 to preserve the strongest domain
- noncoordinate numeric metadata (including `normalized_1000`) never gets a numeric location weight; malformed/ambiguous token segmentation fails closed
- source(original) IDs/JPEG gold, frozen DEV, action grammar, private data and original scorer are **unchanged**. The field weights are hypotheses, not validated improvements.

`h426/source_error_audit.py` and tests give a source-backed diagnosis and field-targeted loss proposal. Since this initial plan, H426's **actual pinned Qwen tokenizer** preflight passed on all 32,000 original H422 source targets, and synthetic PyTorch CPU autograd tests passed. **No actual H426 model training, full image-expanded AutoProcessor qualification, or independent blind OOD validation has occurred.** Compatibility with tokenizer target spans is not model-quality evidence.

## Research guardrails / next qualified engineering steps

1. **Completed (target-tokenizer only):** source-hash-pinned original 32K Qwen tokenizer target-mask preflight. **Not completed:** run the entire Qwen3-VL-2B multimodal image AutoProcessor on a synthetic image with Torch/Torchvision, verifying true expanded image patch prefix, grid, answer boundaries and the new strict target-only mask. Keep original H425 receipts unchanged.
2. Before any **new** training, independently resolve the still-open [source/rendered screenshot ML rights review #384](https://github.com/TheHalfMoon/Hala/issues/384). Owner permission claims do not replace the official H199 admitted TRAIN manifest and independent title/asset review.
3. If authorized, run an original SHA-pinned **PRIVATE FREE GPU** paired/Arabic replay experiment *from authentic H425 LoRA*, not H419 disqualified final. Do not pretend the new model is trained until actual GPU receipts, checkpoint SHAs and original JPEG score replay verify it. Preserve/re-measure 81/144 old, 33/96 fresh, 24/24 original Arabic and slider family as nonregression protections; compare against exactly the current H425, not H417.
4. Treat H421 as **already-exposed author-development data** after this error analysis, never as an untouched/sealed blind OOD. An independent party must custody, preregister and score new unseen bilingual source-family-disjoint tasks as specified in [independent OOD evaluation #385](https://github.com/TheHalfMoon/Hala/issues/385). Do not release/merge/publish model or declare healthcare safety on author-visible research DEV.
5. All original screenshots, private model weights and per-prompt predictions stay on owner-private storage. The GitHub repository contains source code and aggregate counts only.

Evidence from this document can prioritize further engineering but cannot satisfy external rights, independent evaluator, clinical safety, or production release gates.

## Real Qwen3-VL tokenizer original-source preflight (owner CPU proof)

The H426 proposed field-specific numeric loss now has a **real tokenizer** preflight script `h426/real_tokenizer_preflight.py`. It was executed on the authorized Windows machine using an isolated **Python3.11 + Transformers 4.57.3 + Qwen2TokenizerFast**, with Qwen official immutable revision `89644892e4d85e24eaac8bacfd4f463576704203`. It verified original H422 **32,000 original distinct source-native drag action rows**, 10,666 slider, 10,666 slide_resize, 10,668 text_highlight. Exact training row SHA-256 `b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa`. All 32,000 original source supervision action strings were reconstructed from pixel-native source coordinates to H425's normalized-1000 JSON, then tokenized in the real Qwen image-containing chat template using actual source instructions and H425's exact English task suffix. The field-target mask reported **32,000/32,000 successes, 0 rejected**. No GPU, image loading, model weights, PHI, screenshot disclosure or training performed. This is a **tokenizer-span and target-field compatibility result**, not full multimodal-image processor proof, a trained-model quality gain, external OOD, or legal rights admission.

Tokenizer source bytes are pinned locally by SHA-256: `tokenizer.json=a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`; `tokenizer_config.json=c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399`. Read the trusted Qwen tokenizer from its D: private cache snapshot matching the pinned revision; do not let the source verifier silently fetch a mutable tokenizer version. Local snapshot mode avoids a Transformers upstream `_patch_mistral_regex` helper's otherwise unnecessary Hugging Face online model-info call, and avoids needing a paid GPU.

Owner-only reproduction (no GitHub model/data commit, no running model or GPU):

```powershell
$env:HF_HOME='D:\\Hala-H426-Tokenizer-Private\\hf-cache'
$env:HF_HUB_OFFLINE='1'
& D:\\Hala-H426-Tokenizer-Private\\venv\\Scripts\\python.exe h426\\real_tokenizer_preflight.py --rows D:\\Hala-H422-Paired-Private\\h422_genuine_original_paired_32k_actions.jsonl --tokenizer-snapshot D:\\Hala-H426-Tokenizer-Private\\hf-cache\\hub\\models--Qwen--Qwen3-VL-2B-Instruct\\snapshots\\89644892e4d85e24eaac8bacfd4f463576704203
```

The script refuses changed source row SHA, changed local tokenizer bytes, wrong model release/version, invalid original pixel geometry, incomplete full 32K source rows, or any single rejected coordinate target mask. It outputs *aggregate counts only*. New standalone CPU test suite `tests/test_h426_real_tokenizer_preflight.py` provides synthetic drift/falsification coverage. **Current H426 loss remains an untrained research proposal and cannot supersede independently validated H425 until actual rights and GPU evidence gates pass.**


## H426 strict native answer-only supervision (code-only research)

On 2026-11-05, after independent source error evidence, `h426/target_aware_loss.py` began **rejecting unexpected prefix/suffix text** around the native drag JSON target instead of silently weighting the valid JSON embedded in garbage assistant text. It accepts an exact native drag JSON action and the standard Qwen `<|im_end|>` end marker, and rejects any other extra output. Corresponding strict native-target tests are in `tests/test_h426_target_aware_loss.py` and `tests/test_h426_full_processor_contract.py`. These are synthetic research-contract changes only; full original 32,000-case **real-tokenizer preflight must be rerun against this exact new code head** before claiming its new strict gate is qualified. The successful 32K owner run described above was on an earlier H426 commit, before this particular strictness fix. H425 model evidence and H421 results remain unchanged. No paid GPU, H426 training, public weights, actual full image-processor success, independent rights permission, or blind external bilingual OOD is claimed.

## Official Qwen3-VL image-patch cardinality preflight (CPU-only proposed guard)

The optional full-processor contract in `h426/full_processor_contract.py` now verifies more than token existence. Qwen's image processor expands one visual sample to exactly

```text
required_image_pad_tokens = product(image_grid_thw[0]) // (image_processor.merge_size ** 2)
```

for a nonzero grid with exact divisibility. The original complete-head and complete-answer processor output must each contain **exactly** that count of `<|image_pad|>` tokens, and none can occur in supervised target-only tokens. The grid and processor merge size must match across both encoder invocations. The source of this convention is [Qwen3-VL's official Hugging Face processor `replace_image_token`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py); the actual H426 experiment should remain pinned to its specified Transformers 4.57.3 revision, not a mutable latest library. The synthetic `FakeImageProcessor` unit tests now reject a plausible but mismatched enlarged visual grid and refuse a missing image merge size. This strengthens the *planned* full image-processor compatibility check; **do not claim the actual processor and real pixel patch expansion were verified** before a pinned CPU run with PyTorch/Torchvision and a synthetic source image.

### Independent H426 source target uniqueness

The proposed target-coordinate mask also rejects duplicate drag JSON keys even when they parse without an error under default `json.loads`. Otherwise a repeated `start_point` or `end_point` could lead the JSON parser to interpret the last occurrence while field-specific token weighting selected the first. Synthetic regression tests exercise duplicated coordinates and metadata fields. Real 32K original H422 tokenizer supervision needs to be independently rerun on the latest exact H426 branch to count as newly qualified.

## H426 full image-processor exact-byte custody: reuse H218/H387 (research only)

Previously, `h426/full_processor_contract.py::run` checked only `tokenizer.json` and `tokenizer_config.json` before calling `AutoProcessor` directly on a mutable cache. That was insufficient to bind the **actual image/video preprocessor configuration, chat template and vocabulary**: changing an unverified processor config could change image patch expansion without changing the two tokenization hashes.

This source implementation now **reuses**:
- `h218.pinned_qwen_processor_bytes.verify_pinned_snapshot` to check all **eight** original official Qwen3-VL-2B processor file sizes, SHA256 digests, no symlinks/extra files and exact immutable revision `89644892e4d85e24eaac8bacfd4f463576704203`.
- `h387.verified_processor_handoff` to hash the owner snapshot again **while copying** those eight verified public files to an isolated temporary directory, keep that directory alive throughout `AutoProcessor` initialization and all 3 synthetic image task checks, then remove it.
- Strict `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` preconditions, `local_files_only=True`, `trust_remote_code=False`, expected exact `Qwen3VLProcessor` class and H425's `transformers==4.57.3`.

**Actual multimodal preflight has not yet been run** against this new source: CPU-only synthetic provenance/negative tests verify reject paths; an actual offline environment with the eight-file snapshot, `torch`, `torchvision`, `Pillow` and `transformers==4.57.3` must still execute the command below before any claims about real visual patch alignment. This source does NOT load model weights, train, use Kaggle GPU, execute GUI actions or evaluate private/hidden source data.

From the repository root, against an **explicit eight-file verified public processor directory** on a private/free local CPU, rather than a mutable model-cache shortcut:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m h426.full_processor_contract --tokenizer-snapshot 'D:\Public-Qwen-Verified\89644892e4d85e24eaac8bacfd4f463576704203'
```

The supplied example path is illustrative, **not a claim that this directory exists**. Passing source-level tests, the prior original 32K tokenizer-only preflight or GitHub CI alone does NOT qualify the real full image processor, newly trained H426 model, independent blinded bilingual OOD, rights permissions or production release.


## H426/H425 paired source-level loss bridge (2026-11-06; NOT trained)

`h426/h425_loss_bridge.py` now imports the **unchanged H425 generated training implementation** and its `coordinate_weighted_loss`, then calculates the new H426 `proposed_h426_weighted_ce` on the exact **same** model output, assistant-only label tensor and original `case["domain"]`. It returns two differentiable loss tensors for research inspection but has **no optimizer, backward(), checkpoint, dataset fetch, GPU job or model release code**. It rejects incorrectly shaped single-screenshot identities, nonlowercase screenshot SHA, non-EN/AR cases, fragmented supervision suffix, unmasked prompt/image tokens, illegal negative/out-of-range target IDs, mismatched output-vs-label length and nonfinite *supervised* logits. It avoids scanning huge unsupervised Qwen vocabulary logits to stay bounded on a Tesla T4.

It also exposes `original_vs_target_weight_diagnostic(pieces, domain)` for **Torch-free CI**. On a public synthetic native drag JSON target with character-level tokens, the original H425 numeric-token weighting incorrectly also includes the four numeric characters of `normalized_1000`, whereas H426 weights only the real coordinate numeric spans: slider start/end=8; slide_resize end=16; text_highlight both=12. This is a source-level weight-distribution proof only, not a measured improvement on the model.

**Research call site** (only after independently verified real image source, authentic trained H425 adapter/receipt, rights-qualified source, and a controlled GPU setup; this snippet is NOT an executed run):

```python
from h426.h425_loss_bridge import compare_original_h425_to_h426
# H425's original source_image/to_inputs have already verified image bytes and
# constructed real Qwen pixel-conditioned assistant-only labels.
batch, labels = h425.to_inputs(processor, row, source_zips)
output = model(**batch)  # H425 authenticated trained model, read-only comparison
proof = compare_original_h425_to_h426(output, labels, processor, row)
# Optional research logging: aggregate loss values only, NEVER private row.
# Do not call backward(), optimizer.step(), publish weights or leak source rows.
```

**Owner-independent CPU mathematical cross-check**, using hand-constructed synthetic Torch logits with an intentionally difficult drag endpoint (not a model forward, original screenshot, true source label or clinical score), gave original H425 weighted CE 2.608814 and H426 weighted CE **2.837483 for slider, 3.954996 for slide_resize, 3.216238 for text_highlight**. In the synthetic H426 slide_resize objective, the actual Torch gradient was finite and nonzero. These are *toy-example mathematical objective numbers only*, do **not** indicate any observed Qwen accuracy improvement, and are not a run of the complete `h426/h425_loss_bridge.py` on a real source/model batch.

`tests/test_h426_h425_loss_bridge.py` includes Torch-free original/H426 coordinate weight-contrast and source-provenance negative tests (runnable under existing GitHub CI), plus optional actual differentiable Torch comparisons and strict label/finiteness negatives. Existing GitHub CI has no Torch and transparently skips actual Torch tests; this must **never be reported as real H426 trained-model qualification**. The sealed external bilingual OOD custodian (#385) and independent rendered-screenshot/derivative-weight TRAIN rights review (#384) remain unresolved, H199 admitted manifest allowlist is empty, and H426 has not trained a model.


## Reproducible H425-versus-H426 CPU loss proof (no GPU or private data)

The owner-private H426 research branch includes `h426/probe_cpu.py`. On a CPU environment with PyTorch installed, from the repository root run:

```bash
python -m h426.probe_cpu
```

The script imports H425's actual frozen objective from `h425/kaggle/h425_h417_replay_paired.py` and H426's proposed objective through `h426/h425_loss_bridge.py`, applies both to exactly the same **invented character-tokenized drag action and synthetic logits**, and tests that all resulting CPU gradients are finite and nonzero. There is no network access, no dataset read, no Qwen model weights, no optimizer, no update, no Kaggle kernel and no GPU. The program fails closed if the source-loss comparison or gradient evidence drifts. Its JSON result reports explicit `train_steps=0`, `source_images_or_model_weights_read=false`, and `independent_ood_accuracy_measured=false`.

On a separate assistant CPU environment (`torch 2.10.0+cpu`), a source-equivalent local replay of the exact H425/H426 loss functions yielded:

| Invented task | Frozen H425 CE | H426 CE | H426 gradient abs sum |
| --- | ---: | ---: | ---: |
| slider | 2.608814 | 2.837483 | 1.443396 |
| slide_resize | 2.608814 | 3.954996 | 1.552368 |
| text_highlight | 2.608814 | 3.216238 | 1.480329 |

The `h426/probe_cpu.py` script also passed its initial source-equivalent local CPU execution. **This is a toy mathematical objective and gradient test, not a real owner Qwen3-VL model forward, not trained H426, not new original source-label evidence and not an accuracy measurement.** The optional `test_reproducible_cpu_probe_three_domains_no_model_or_gpu` in `tests/test_h426_h425_loss_bridge.py` runs this probe only when Torch is available; source-only GitHub jobs without Torch transparently skip it. A passing standard 22-workflow CI therefore cannot be used to claim that Torch CPU gradients ran on that exact GH Actions head.

The actual 32K original H422 real tokenizer replay must still be repeated on the final strict H426 loss version. Only then consider a new H426 experiment from the genuine owner-private H425 trained adapter in an authorized free GPU environment, with independently approved rendered-screenshot TRAIN rights and an independently custodied, pre-registered bilingual source-disjoint blind evaluation. Neither rights issue #384 nor OOD issue #385 is satisfied by this source-only probe.

 
## First-party invented bilingual GUI geometry generator (2026-11-06)

`h426/synthetic_drag_curriculum.py` adds a reproducible **offline Pillow-only** image and action generator for the H426 geometry research domains `slider`, `slide_resize`, and `text_highlight`. It does not download external GUI screenshots, reference patient data, or read private original H60/DragOn labels. All on-screen visual elements (generic cards, rails and words) are drawn from scratch and every generated screenshot has exactly **two different source-native drag actions**, with English and Arabic user instructions, preserving original 960×640 pixel coordinates as well as normalized `0..1000` JSON targets. Different invented visual template families are reserved for train and dev; this is NOT independent, sealed real-world OOD.

The generator is lazy and does not store 24,000 screenshots in the repository: callers explicitly request `generate(split, index, domain)`, receiving in-memory `PIL.Image`, PNG bytes, SHA256 digest and paired labels. All rows use `source_role=firstparty_generated_synthetic_research_not_admitted`, intentionally **not compatible with implicit promotion to official H199 TRAIN admission**. It does not bypass Hala's rights review or its original source provenance; any future source-authorized training runner must separately validate media/labels and approve the role before using generated source bytes.

Reproducible lightweight CPU audit from repository root:

```bash
python -m h426.synthetic_drag_curriculum
python -m unittest discover -s tests -p 'test_h426_synthetic_drag_curriculum.py' -v
```

The initial local Pillow 12.3.0 audit produced **384 distinct generated PNGs and 768 EN/AR native drag action rows** (64 images per split×domain), no source-image SHA collisions and disjoint invented family IDs. An earlier prototype collided on visually identical screenshots due a repeating workspace index; it was fixed **before publication** by visibly encoding a unique bounded image index, then the 384-image audit passed. GitHub CI now has source-only tests of repeatability, SHA, normalized geometry, both-prompt target masks, family-disjoint subsets and the non-admitted training role. This is a **data generator and source-contract test only**; no real H426 LoRA model was trained, no model-weight inference or actual learned accuracy was measured, and original rights issue #384 and independent real bilingual OOD issue #385 remain open.
 

## H426 actual H425 visual-input/target bridge (2026-11-06)

New `h426/synthetic_h425_processor_bridge.py` uses the existing **first-party invented UI generator** without exporting its images to GitHub. It reopens and SHA-256 checks the actual in-memory source PNG, requires 960×640 RGB PNG media and an explicit **not-admitted** source role, resizes the processor-bound image to **600×400** using the **exact H425 LANCZOS thumbnail algorithm** (H425 keeps original 960×640 dimensions for target normalization), imports the frozen original H425 `target` and `prompt` functions verbatim, and verifies that their canonical native-drag JSON gold equals the generated coordinates. English and Arabic versions of each synthetic PNG must have distinct instructions and source targets, and share identical source image bytes.

`pack_generated(split,index,domain,language)` returns in-memory processor-ready image, prompt, exact canonical JSON and aggregate provenance. It has **no model weights, GPU, optimizer or download code**, and sets `source_is_admitted_to_train=false`. `check_generated_with_real_processor` calls the H426 image-patch/assistant-only token-supervision `check()` on these actual generated UI images. The pinned eight-file offline Qwen `h426.full_processor_contract.run` now includes **12 additional synthetic GUI cases** (train/dev × 3 geometry domains × EN/AR), alongside its original **6 gray-image cases**, all within its H218/H387 isolated immutable processor snapshot and with **no base-model weights**. The actual on-device pinned full Qwen3VLProcessor run **has not been executed on the latest H426 source**, so do not claim 18/18 real processor success until a genuine provider receipt exists.

`tests/test_h426_synthetic_h425_processor_bridge.py` provides CPU-only deterministic SHA/PNG/resize, frozen H425 prompt/coordinate parity and image identity, plus fake Qwen vision-boundary integration and adversarial source-role/PNG/target mutation refusals. **Fake processor tests are not actual Qwen pixel-patch tests**. Source-only GitHub CI will qualify the Python contract; actual full-processor qualification needs original eight pinned PUBLIC processor files, offline flags and Torch/Torchvision on an authorized free CPU device.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -m h426.full_processor_contract --tokenizer-snapshot 'D:\Public-Qwen-Verified\89644892e4d85e24eaac8bacfd4f463576704203'
```

The path is illustrative only. Independent original source TRAIN copyright review #384 and genuinely unseen EN/AR external OOD #385 remain open. This research bridge does not authorize using old H425 copyrighted source assets for new TRAIN, and does not prove a H426 fine-tuning gain or production release.


## Invented DEV paired metric and optional read-only T4 inference

`h426/synthetic_pair_eval.py` scores **only first-party generated DEV cases** using the original frozen H425 `parse_drag`, `score_predicted` and `TOL_NORM` rather than inventing a more favorable H426 success metric. A synthetic image has two different paired instructions (EN and AR), each with a verified screenshot SHA. The scorer rejects missing, duplicated, swapped and malformed predictions, reports correctness for **both actions on the same image**, and separates start-only, end-only and both-point errors. It accepts caller-supplied predictions and **does not run or load any model**. A baseline/candidate comparison always uses the exact same synthetic image and targets. These DEV families are *author-visible synthetic research samples*, not independent blind source-family-disjoint OOD.

`h426/synthetic_inference_eval.py` is an **optional read-only research-only** bridge for a model and a verified Qwen3VLProcessor already loaded in a separate owner-controlled FREE Kaggle T4 process. It does not download/transfer model weights, instantiate a model, train, change parameters or persist raw predictions. It can run 1–32 distinct invented DEV images *per domain*, for 6–192 EN/AR actions, and returns only per-domain and per-language aggregate metrics. It requires `HALA_H426_SYNTHETIC_READONLY_EVAL=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, a live owner Kaggle kernel on a Tesla T4, and a preloaded CUDA model plus official Qwen3VLProcessor; it fails closed outside those preconditions. **The module does NOT verify model/processor byte SHA itself; the caller must use H425's independently verified private adapter and H218/H387's pinned public processor loader for credible provenance.**

Usage inside an **already-authorized private owner GPU environment**, with those objects prepared and verified separately:

```python
import os
from h426.synthetic_inference_eval import evaluate_already_loaded_model
os.environ["HALA_H426_SYNTHETIC_READONLY_EVAL"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# model and processor must ALREADY exist as authenticated local objects.
# This is an API example, not a command to fetch or upload private weights.
research_report = evaluate_already_loaded_model(model, processor, count_per_domain=4)
# Only aggregated scores; no original screenshot bytes or raw predictions.
```

Source-only GitHub tests `test_h426_synthetic_pair_eval.py` and `test_h426_synthetic_inference_eval.py` prove scorer equivalence, paired identity, endpoint-only error categories and zero-leakage/zero-GPU preconditions. **No T4 model inference was executed on H426 here**, and no observed improvement over H425 has yet been measured. The only genuine trained model remains H425 on original author-visible source DEV, with unresolved separate screenshot rights #384 and externally custodied sealed EN/AR OOD #385. Do not promote invented DEV into independent OOD and do not report any baseline/candidate synthetic metric until actual authenticated model-generated predictions exist.


## Synthetic generator v4: visual grounding and anti-shortcut controls (2026-11-06)

The current **`hala.h426.firstparty-invented-bilingual-paired-drag.v4`** schema intentionally supersedes earlier v1/v2/v3 versions, which had unstable label semantics or overly simple language/position shortcuts. It changes the generator to vary slider rails and targets, resize cards, and highlighted text positions and actual visible words across source images. All target endpoints must remain visible on the rendered source image. For the two distinct actions per image, `en` and `ar` are assigned to different visual targets using a deterministic SHA-256-derived mapping with a private-source-independent fixed salt. This prevents the model from treating English as *always object #1* and Arabic as *always object #2*, or using simple parity of the image's visible workspace index. The instructions refer to whichever labeled visual object is assigned, never to a fixed language-specific position. These are **anti-shortcut measures, not evidence of learned multilingual grounding**.

`tests/test_h426_synthetic_drag_curriculum.py` now checks that the target assignments are reproducible and approximately balanced by language over bounded sample windows, text instructions match actual rendered words, target pixels are visibly marked, and source coordinates/words vary across dozens of images. **Important experimental provenance:** the previously recorded local **384 generated PNG / 768 action** audit was conducted on an *earlier generator revision* and must **not be represented as a completed audit of v4**. An exact-v4 local CPU audit was completed on 2026-11-07 using `audit_balanced(64)` on the authorized Windows device: **384 distinct in-memory generated PNGs and 768 paired EN/AR actions**, 64 actions per split/domain/language, without duplicate image SHA-256s or TRAIN/DEV invented-family overlap. This is a generated-image/source-contract audit only, not a model, processor, GPU, sealed evaluation, or source-rights result. The official pinned Qwen processor token-mask replay on the final v4 source remains pending; GitHub's source-only unittest coverage does not replace that. H425 training remains immutable and no new H426 model training, real-model accuracy or independent OOD has occurred. The generated examples remain **research NOT ADMITTED** to the official H199 TRAIN source registry.
