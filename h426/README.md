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
