# Hala H25 — actual saved Kaggle DEV baseline, Version 1 (2026-10-18)

**Status: genuine model measurements, but UNQUALIFIED. This does not authorize any Hala or Orcel execution, model adoption, training or release.** All artifacts below were copied intact from an actual completed private Kaggle Saved Version and an independent local CPU replay. Historical H22/H24 TEST artifacts and H24 V1/V2 archive files remain unchanged.

## Runtime provenance

- Private Kaggle actual, platform-resolved URL: https://www.kaggle.com/code/abdulazizshehri/hala-h25-first-bilingual-mind-dev-baseline
- Saved Version **1**: Kaggle CLI push reported success; later `KernelWorkerStatus.COMPLETE`.
- Kaggle title/slug warning at initial submission: `kernel-metadata-source.json` requests ID `abdulazizshehri/hala-h25-mind-dev-first-20261018`, but Kaggle created the actual title-derived slug shown above. The source metadata file is preserved **without rewriting history**.
- Source: Hala Draft PR #8 exact signed+DCO SHA `f24ca699b9b4076e16ff85d02eb8ffbddc497131`. GitHub had 4/4 source CI checks passing before this submission.
- Model `Qwen/Qwen3-VL-2B-Instruct`, exact upstream revision `89644892e4d85e24eaac8bacfd4f463576704203`, **FP16**, no LoRA, no tuning, no quantization or changes to weights.
- Kaggle-runner reported **Tesla T4**. Model load 28.474920 seconds; `torch.cuda.memory_allocated()` after load 4,256,128,512 bytes; `torch.cuda.max_memory_allocated()` peak 4,379,026,944 bytes. These are **reported allocator counters**, not full device memory, validated hardware, energy or cost readings.
- Public H25 original first-party synthetic source SHA256 `b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b`. H25 V2 model-input projection SHA256 `cceed2575f3f01995d489c811e2ee38f32272492fc2b533c0197ec0ba27c8ccd`. Strict decision protocol `h25-seven-pair-dev-first-v1`.
- No data entered a physical Orcel executor. No model output was treated as a permission grant, no personal/sensitive records supplied, and no billable GPU tier was deliberately selected. No proof of external account charges was generated.

## Actual measured result, recomputed from raw

| Measure | English | Arabic | All |
| --- | ---: | ---: | ---: |
| Actual DEV case responses received | 21 | 21 | **42** |
| Model inference exceptions | 0 | 0 | **0** |
| Strict, fully executable-schema-valid Hala typed proposals | 0 | 0 | **0** |
| Strict tool correct / exact decision correct | 0 | 0 | **0** |

All **42** raw outputs are syntactically parseable JSON, but none conforms to the required `{"kind":"next","tool":"...","subgoal":"..."}` contract. The model often treats `kind` as its intended action (`visual`, `observe`, `clarify`, `abstain`) instead of the required constant `next`; `tool` may be another action, the target identifier or a malformed category; `subgoal` is sometimes prose instead of the allowed normalized value. **No amount of JSON syntax validity makes these outputs trusted actions.**

Post hoc non-authoritative study: **28/42** raw `kind` values happen to agree with the *expected tool*, **17/21 English and 11/21 Arabic**. This violates the actual schema and is NOT valid model success or permission; do **not** transform it into an accepted result, inflate strict scores or compare it to H24 as if the same protocol/fixtures were used. For the six `unsafe_user` cases, four raw `tool` fields were `visual` even though none of the `kind` fields was `visual`. No native action was executed. Never infer a safe refusal from the untrusted field combination. See `posthoc-diagnostic.json`.

Source content is public and limited to 42 related, first-party synthetic **DEV** tasks (21 Arabic, 21 English; exactly six cases per state across seven states), **not a truly unseen TEST**, deployment, HalaType visual evidence or reliable multilingual closed-loop action success. The test corpus was designed after viewing H24 DEV errors, so no holdout or causal generalization claim is made.

## Verified immutable raw evidence

| Exact archived file | SHA256 |
| --- | --- |
| `result.json` — Kaggle-measured raw 42 outcomes | `2fe2a558f479b255a6b10df3d01344815da57fec1fadcb2e5457a58e5b01028f` |
| `independent-audit.json` — CPU exact-source/raw replay | `b2752851da5d7787e53ef457bef19b066106385d0a2def19cfac946588fd0676` |
| `posthoc-diagnostic.json` — descriptive output distribution, NOT strict scorer | `f9535597e74c7d87971ad4e98d3f0468f35938124e694f850f0dce748c59081e` |
| `notebook-source.ipynb` — original submitted source notebook | `8b57e5498cfc0c6e7bfed8b326540d90685d97ddf0040843471843f2897822e0` |
| `kernel-metadata-source.json` — submitted metadata with original requested slug | `085cbfe559ec3d13a5ee9d303d3dba86667223131fcf0113d75f6edeecc2d808` |

Raw `result.json` was downloaded using Kaggle CLI from the actual saved-version slug into a fresh local folder, SHA256 recorded *before* running the independent verifier, and its untouched bytes copied into this evidence branch. Replayed via:

```bash
python -m tools.verify_h25_dev h25-mind-dev-v1/result.json \
  --expected-sha256 2fe2a558f479b255a6b10df3d01344815da57fec1fadcb2e5457a58e5b01028f \
  --source-commit f24ca699b9b4076e16ff85d02eb8ffbddc497131 \
  --output NEW_UNEXISTING_INDEPENDENT_AUDIT.json
```

The verifier runs with Hala source at exact `f24ca699...` and source-frozen independent case labels, not model weights. It confirms all 42 outcome identities/order, SHA-pinned input, model/protocol/version/DEV status, raw-vs-scored correctness and language/state aggregates. **This is a separate CPU run of the same algorithm, not independent expert adjudication of the oracle or a hardware attestation.**

## Decision

**No adoption of this unchanged base model as a strict HalaMind output generator.** Its typed contract validation is zero on these DEV conditions. Do not simply reinterpret `kind` or promote `tool` without a *new prospectively defined schema*, threat review, a new comparison protocol, independent bilingual adjudication and a truly separate held-out benchmark. First identify instruction/serialization behavior and representational limits without tuning repeatedly against this same exposed DEV dataset. Evaluate native permissions only under Orcel's own governance; Hala cannot execute actions.

The missing independent PStack/security review, rights/legal training clearance, actual H23 pending-CUDA cancel/restart, reliable Arabic Type/RAQM visuals, native Orcel integration and multi-step verified outcome benchmarks remain open. No PR merge and no release.
