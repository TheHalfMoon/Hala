# H160 — verified real external-app native model outcome (2026-10-27)

**Completed but NOT promoted. Native original-app task success: H156 0/16, H157 0/16.** This is an honest negative cross-application transfer result, despite successful H157 same-app MiniWoB++ DEV improvement to 60/72.

## Frozen provenance

- Same original MIT `tastejs/todomvc` application distinct from the MiniWoB++ training app, pinned upstream `ff43b02e59dfa604386bb382034b2cd07c2bcd8a`; new H159 cases 301–316, not H158 consumed cases.
- New stable H159 Chrome images: **16/16** exact independent replay SHA matches before inference; source SHA `ac3dd44a65349223298b44470fce9dde8de78541f2b5997b003bf72a08ca13c7`.
- Frozen H160 bilingual image+instruction SHA `c7521a6e0f359867846c15bbfabb06b1a09aa117d93d4a3b5958072d6bec92fc`.
- Prior H156 LoRA SHA `732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6`; new trained H157 LoRA SHA `181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8`.
- Private Kaggle Nvidia T4 completed **32/32 genuine frozen-model inference calls**, 16 per adapter; **32/32 valid single-click JSON**. Zero optimizer steps; no paid compute, no H140 sealed blind calls.
- Independent one-shot owner Windows Chrome replay executed **32/32 model actions** in separate original UI contexts with exact frozen input images; actual application UI state was checked after each click. **Zero task completion for either adapter** in all three task types (8 named-checkbox, 4 Active filter, 4 Clear completed), balanced across Arabic/English. 0 gains, 0 regressions, 16 both-fail, exact paired sign p=1.0.
- Original app stores state in memory, not across page reload. Results score actual in-session state, with no false persistence claim.

## Posthoc geometric diagnostic — descriptive only

Using private reference click centers *after* the frozen evaluations, H156 mean Euclidean center error was **226.3 px**, H157 **139.8 px** (medians 224.3 px vs 149.1 px) across matched 16 images. All 32 predictions were **more than 50 px from source target centers**, hence the lower error still did not produce usable browser task success. These point distances are exploratory; target geometry and successful click area differ, and they must NOT be reported as accuracy or used to tune/retry this consumed cohort. The H160 trial is fully consumed and locked.

## Decision

**No promotion of the H157 adapter as cross-app-capable; no model release.** Hala flagship 35B untouched. The next phase must collect new, genuinely varied **original app interfaces** from multiple codebases, train on TRAIN-only fresh source without H160 or H140 leakage, and reserve distinct unseen app/scene BLIND sets for independent evaluation. Focus on cross-app button, checkbox, anchor and layout grounding rather than simply repeating MiniWoB++ optimizer steps.

Owner-private receipts: `private-h160/kaggle-output-v1/h160-free-t4-inference-receipt.json`, `private-h160/H160_32_ORIGINAL_APP_MODEL_NATIVE_TASKS.json`, and `private-h160/H160_GEOMETRY_POSTHOC_ONCE.json`. Never commit individual media, model actions, reference coordinates or weights.

No paid models or GPUs, H140 reuse, main merge, publication, or release.
