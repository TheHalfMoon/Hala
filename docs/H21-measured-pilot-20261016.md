# H21 actual shared pilot — 16 October 2026

The first fixed shared Mind/Type pilot completed 116 steps and its original small adapter is retained. Fresh free CPU-session public retrieval and finite tensor decoding passed. **Actual pinned-base PEFT attachment and frozen-task candidate quality remain unverified.** Colab refused free GPU allocation after a usage pause; the original runtime expired only after the trained bytes and baseline evidence were published.

| Frozen strict outcome | English | Arabic | Total |
|---|---:|---:|---:|
| Unchanged-base development tasks |0/4|0/4|0/8|
| Unchanged-base test tasks |0/4|0/4|0/8|
| Development episode Mind semantic correct |0/32|0/32|0/64|
| Test episode Mind semantic correct |0/32|0/32|0/64|
| Standalone Mind exact correct (10 unique cases) |0/5|0/5|0/10|

The standalone cases repeat in both runs and are not 20 independent cases. Independent audits passed 1,124 development and 1,084 test consistency checks, including persisted final states and raw-call bindings; all ten actual Type PNGs were independently hash-checked. No edit, move or save occurred. Development errors were predominantly invalid/wrong-kind Mind outputs. The frozen unsafe_proposals counters count six development and four test protocol rejections: these are not ten confirmed harmful/external proposals. Recovery was never reached. No prompt/parser repair or test-driven tuning changed these scores.

The fixed training run used 116 original train-only Mind/Type rows (58 per language), excluding Reliance and all held-out families. One authored epoch, seed42, batch1, AdamW5e-5, rank4/alpha8/dropout0, FP16 base with 112 F32 language q/v LoRA tensors (802,816 scalars). Recorded losses and gradient norms are finite for all steps; the independent artifact audit decoded every scalar as finite. Numerical update logs do not independently reconstruct discarded gradients or establish quality. Training wall time was 116.095s. The original historical training result says trained_pending_durable_archive and remains unchanged; subsequent archival/retention evidence is separate.

## Immutable identities

- Development/training source: `5b6eb4bf29acbaf21aa199f113f78af4f268b3b7`; all four exact-head CI checks passed. Last full local integration:216 tests, one Windows symlink-privilege skip; dependency repair:18 scoped tests plus independent delta review.
- Base evaluation source: `6b9bdff139d57ca29d0e4c5774c2d71805e8188e`; runtime SHA256 `da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35`. Subsequent dependency repair left this runtime and frozen data/settings unchanged.
- Original synthetic freeze:120 PNGs/136 rows; manifest SHA256 `ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e`; ZIP SHA256 `d6be1ffba54a3a44978da4c922130fc2d7e74bdc2532a14d0d1e7cc3a691cead`, retained at `5a4266a57a332165b57c2310ee8219c99b2bcfe0`.
- Actual baseline and new-schema pretraining probe: signed archive commit `c117f1d30e43aadcb4e9842f421c9029141a584e`. Public Colab retrieval verified original baseline/probe hashes before training. Development baseline SHA256 `e9f98694089d5411d3a6c570c7de2ede8f82cbeb3132e47904d17301030c0768`; test baseline SHA256 `112d7c09a5d6cabbf68d8d74518c98b1f5d53d114ddc9c58dd5cf65c59bb2ad4`.
- Original trained bundle: signed archive commit `274024e14699599a51fca2c948af40819532e18c`; manifest SHA256 `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d`. The safetensors file is3,227,648 bytes, SHA256 `7eb2f17df11151e5b77a5c715c59f56c854ca7b0d0da3ee591f394b501826fa7`. Original PEFT config/metadata were preserved; no optimizer pickle or base weights were exported.

Fresh CPU Colab Python3.13.16 had no old training files. It retrieved the original bundle and complete freeze from immutable public commits, checked external full identities/hashes and decoded112 F32 tensors/802,816 finite scalars. This establishes retention and safe-format integrity, not model/PEFT loading or task quality. No local large model weights, paid services, Aladdin edits, main merge or production release occurred.

Genuine PStack independent combined/security reviews, Jev advisory judgments and Graft source-graph checks are recorded per source phase. Graft deep_context_built=false is not sandbox/correctness evidence. Gemini3.5Flash assisted routine Colab cells; root corrected API/launch errors before execution. An actual safetensors0.6.2 import failure was retained and repaired to the officially required0.8.0 through independent review; CPU tests alone had not exposed that compatibility defect.

Next: qualify an eligible free GPU environment, load the retained bytes on the pinned base and compare identical frozen tasks. A Kaggle evaluation-only guard is being developed separately after user authorization, with the original Colab-only training guard retained. Preserve no-improvement results; no new adapter or precision is adopted from this pilot. Representative precision/cost, expanded serving lifecycle and native integration remain incomplete. The founder controls PR/main/release.
