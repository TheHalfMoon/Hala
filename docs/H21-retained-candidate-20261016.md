# H21 retained adapter: actual fresh GPU comparison

Updated 16 October 2026. This extends the [earlier retention checkpoint](H21-measured-pilot-20261016.md); that checkpoint's pending-quality statements describe its earlier observation time. No historical result was rescored or replaced.

The original single shared 116-step Mind/Type LoRA was downloaded in a genuinely fresh free Kaggle T4 session, attached to the pinned unchanged base, and evaluated on the frozen H19 tasks. Complete development success increased from 0/8 to 1/8 and test success from 0/8 to 2/8. All three successes were English. These small, dependent synthetic samples demonstrate a pilot gain, while Arabic completion remains 0/4 in each split. No model adoption, representative generalization, native execution or release qualification follows.

| Strict metric | Base DEV | Adapter DEV | Base TEST | Adapter TEST |
|---|---:|---:|---:|---:|
| Complete persisted tasks |0/8|1/8|0/8|2/8|
| English complete |0/4|1/4|0/4|2/4|
| Arabic complete |0/4|0/4|0/4|0/4|
| Semantically correct initial plans |0/8|8/8|0/8|8/8|
| Episode Mind semantic correct |0/64|26/59|0/64|26/56|
| Standalone Mind exact correct |0/10|3/10|0/10|3/10|
| Action calls |6|54|4|55|
| Action errors |6|42|4|42|
| Environment failures / recoveries |0/0|1/0|0/0|1/1|
| Episode model errors |64|5|64|0|
| Protocol-rejected proposals |6|0|4|0|
| All model calls |80|123|78|121|

Protocol rejections are included within action errors; do not add those columns. Episode model-error counts exclude separate initial-plan and standalone-probe scoring. Episode denominators depend on reached states and are not a matched fixed-question accuracy comparison. Ten standalone probes repeat across DEV and TEST, were already exposed during DEV, and are not twenty independent or untouched TEST probes. Their adapter scores are English 2/5 and Arabic 1/5. Zero unnecessary-call counters do not mean no repeated erroneous actions: such actions are included in action errors. The recovery is an actual synthetic simulator save-failure/observation/retry, not a native application recovery.

## Fixed inputs and model identity

Base: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`. Original training source: `5b6eb4bf29acbaf21aa199f113f78af4f268b3b7`. Actual Kaggle comparison source: `11a3a5359afa896787198e0492ccd036d0b6f4f8`, runtime SHA256 `cbc80167c80461a267a7232b0596ccf176c0c118dbf7772128a92798ee0fca2e`.

H19 contains 136 training rows and 120 frozen PNGs across four train, two DEV and two TEST application families, with English/Arabic and normal/transient-failure variants. The fixed training subset contains 116 Mind/Type rows, 58 per language; Reliance and held-out families were excluded. Families share authored workflow templates. Manifest SHA256: `ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e`. All 120 original PNGs reproduced byte for byte on Kaggle before evaluation. Dataset and parser, prompts, scoring, task budgets and held-out families were unchanged.

The original adapter remains at signed artifact commit [`274024e14699599a51fca2c948af40819532e18c`](https://github.com/TheHalfMoon/Hala/tree/274024e14699599a51fca2c948af40819532e18c/h21-shared-pilot/trained-adapter). Manifest SHA256 `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d`; 3,227,648-byte safetensors SHA256 `7eb2f17df11151e5b77a5c715c59f56c854ca7b0d0da3ee591f394b501826fa7`. Fresh loading checked all 112 tensor names/shapes, 802,816 finite F32 scalars, exact loaded values and frozen parameters before setting adapter_loaded=true. No retraining or adapter rewrite occurred.

Kaggle used Python3.13.15, Torch2.11.0+cu128, Transformers5.16.1, PEFT0.20.0, safetensors0.8.0, accelerate1.12.0, Pillow11.3.0 and RAQM0.10.1. The first attachment failed before any candidate call because preinstalled optional TorchAO0.10 was incompatible with PEFT0.20. Removing that unused optional package restored the original training environment's absence of TorchAO; base weights and required pins were unchanged. This failure is retained separately. Genuine Kaggle's inherited Colab marker was preserved and the reviewed evaluation-only provider guard admits that real environment; the training provider guard remains unchanged.

## Durable actual evidence and independent checks

All paths below are under `h21-shared-pilot/` in the named immutable artifact commits.

| Artifact | Commit / archive SHA256 |
|---|---|
| Kaggle base DEV, 98,107 bytes | `9c1a3c924edfcfee3e422dbb6921bd01301bbacc` / `20b30a46c25dd04b7577785afc291166f987cec87961fd8409b15a4d9dd13be2` |
| Kaggle base TEST, 73,146 bytes | same commit / `6542d126ca6732843f103d0ddee1a31dad4069e12397a546c7907b56ae596f4a` |
| Failed attachment, 11,879 bytes | same commit / `3997fa07312496d8090545940c3ec8eba367f0cc0d53b6d09381f41c07294028` |
| Fresh adapter DEV, 263,482 bytes | `ef359cabe7aefd7d07fc6485df7b68b55090983f` / `0619aa121f044e26fbe1be8533f3ade5c2279c333a0b09bf70ba2448d51bfb25` |
| Fresh adapter TEST, 295,810 bytes | same commit / `be538578cd0e0f3eb6aef4155d9e6a0625a04169ce32d3fefb1bae4cf6649c1c` |

Raw controls are in `kaggle-controls-20261016/`; candidate records, audits and independent reviews are in `fresh-gpu-candidate-20261016/`. Public retrieval round trips verified retained ZIP hashes. Candidate result JSON SHA256 values are DEV `ab57e72fc5efc9badb394fe11272e6896713c2481896630e25418e3e8e4b1b15` and TEST `1de1a9f270b5dfa730c04ef258ceb4f26aabc2a95610ebbaef2d8a1f9ae19831`.

Independent standard-library consistency audits passed 1,124/1,084 base and 2,012/2,001 candidate DEV/TEST assertions. A separate reviewer replayed all 109 candidate action proposals, geometry, state transitions and persisted outcomes: 949 DEV and 955 TEST checks passed. TEST had 54 byte-matched frozen screenshots and one legitimate newly reached state image whose hash/signature/dimensions were checked; that additional image was not independently rerendered. These saved-record audits do not independently attest GPU hardware or reconstruct training gradients. Separate GPU allocations make latency comparisons descriptive.

DEV-only analysis attributes its 42 action errors to 16 Arabic field-point misses with correct text, 12 repeated English drags after the move was complete, seven premature Arabic save hits and seven premature save misses. All 22 typed strings were correct. Five model errors were invalid visual/inspect_failure pairs; another 28 semantically wrong valid Mind calls brought total wrong episode Mind calls to 33. Grounding and state-aware action/recovery are the next data targets; no TEST-driven tuning or new training run was performed. Analysis and reproducer are retained at `d5b5a772426b3cd40de701ee8e9a749d6121540c`.

## Next gates

H22 representative FP16/int8/NF4 source is reviewed and CPU-tested. Published source `07591f109b6683c9302fcf890259ddf0bde91494` passed all four exact-head CI jobs; the initial shallow-checkout historical-source test failure was repaired by fetching full CI history, without weakening assertions. H22 attempt1's ephemeral session expired during a user/usage pause: zero precision result files were retrieved, so its outcomes are unknown. Its prospective manifest and launcher remain retained. Attempt2 requires new hardware qualification and one-profile-at-a-time verified durable exports. No H22 quality/cost outcome or precision adoption is established.

Expanded private serving lifecycle source has CPU tests but still requires independent final source review and actual remote GPU qualification, especially cancellation while real CUDA work remains pending at receipt. Native integration, representative Arabic success, production cost/security and founder merge/release decisions remain open. No Aladdin changes, paid compute, billing activation or large local base-weight downloads occurred.
