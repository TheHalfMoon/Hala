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

`h426/source_error_audit.py` and tests give a source-backed diagnosis and field-targeted loss proposal. The loss helper has synthetic tokenization tests only; **no actual tokenizer/T4 training run or validation yet**. Do not promote the pure helper to an experimentally qualified objective without real Qwen3-VL tokenizer supervision tests.

## Research guardrails / next qualified engineering steps

1. Verify proposed masks against actual Qwen3-VL-2B tokenizer and synthetic labeled examples. Keep numeric text token offsets fail closed; preserve full H425 training receipts.
2. Before any **new** training, independently resolve the still-open [source/rendered screenshot ML rights review #384](https://github.com/TheHalfMoon/Hala/issues/384). Owner permission claims do not replace the official H199 admitted TRAIN manifest and independent title/asset review.
3. If authorized, run an original SHA-pinned **PRIVATE FREE GPU** paired/Arabic replay experiment *from authentic H425 LoRA*, not H419 disqualified final. Do not pretend the new model is trained until actual GPU receipts, checkpoint SHAs and original JPEG score replay verify it. Preserve/re-measure 81/144 old, 33/96 fresh, 24/24 original Arabic and slider family as nonregression protections; compare against exactly the current H425, not H417.
4. Treat H421 as **already-exposed author-development data** after this error analysis, never as an untouched/sealed blind OOD. An independent party must custody, preregister and score new unseen bilingual source-family-disjoint tasks as specified in [independent OOD evaluation #385](https://github.com/TheHalfMoon/Hala/issues/385). Do not release/merge/publish model or declare healthcare safety on author-visible research DEV.
5. All original screenshots, private model weights and per-prompt predictions stay on owner-private storage. The GitHub repository contains source code and aggregate counts only.

Evidence from this document can prioritize further engineering but cannot satisfy external rights, independent evaluator, clinical safety, or production release gates.
