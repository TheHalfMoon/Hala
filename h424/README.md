# H424 — Genuine H419 intermediate checkpoint comparison

Status: PRIVATE SOURCE-QUALIFIED MODEL CHECKPOINT DIAGNOSTIC STAGED. GPU RESULTS NOT VERIFIED.

## Grounding: original H419 completed but failed research gate

The owner's private H419 Kaggle kernel completed 9,452 genuine LoRA optimizer steps on original 30,000 DragOn training screenshots and 976 Arabic original images: 37,808 real image-conditioned passes. Its final original provider receipt SHA256 is f8c491d8522a8349974d05318fded7cbe0349478be27efcbb118a8efdcfa37ac, and the original 12,878,000-byte trained adapter SHA256 is 96d07b07cca54d15e9ce11da4d26fa3748564f09a2974d9a0b48e419ab03eb8c.

Owner Windows replay independently reopened source JPEGs and verified all original 240 DEV predictions/metrics; adapter bytes and original receipt sidecar were separately SHA checked. The result **must not be promoted**: overall correct 94 -> 101 / 240; fresh original 96 cases 18 -> 30; old H417 144 cases 76 -> 71, BELOW frozen >=72 nonregression minimum. H419's own research gate is FALSE, as is release approval. Original H419 was not an independent external OOD test.

## H424: actual checkpoint model experiment, no new training

The same original Kaggle H419 run wrote genuine LoRA checkpoint adapters after steps 1,500, 3,000, 4,500, 6,000, 7,500, and 9,000. All six + final were downloaded from the owner kernel, had real compatible safetensors headers, exactly 12,878,000 bytes each, distinct SHA fingerprints, and were bound to the original owner log's every 150-step authenticated progress line. The private stage contains 7 model snapshots in a SHA-pinned ZIP, with a private manifest, NOT in GitHub.

The new script uses the original frozen H419 source image and evaluation helpers, the original nine source ZIPs, original source-disjoint 240 DEV partition, frozen base Qwen revision, and real original H417 parent. It evaluates the final 9,452-step adapter FIRST and demands the original exact 240 score replay. Then it evaluates the six intermediate snapshots. The research gate is NEVER weakened: correct >18 / 96 on new DEV, correct >=72 / 144 on old DEV, all 240 predicted outputs valid.

If an intermediate checkpoint passes, H424 will label it ONLY as an author-visible internal checkpoint-selection candidate, not as an independently OOD-validated or released model. If none passes, return NO SOURCE-ONLY CANDIDATE. H421's reserved 384-case evaluation, externally disjoint bilingual OOD, source rights and reviewer signoffs remain open.

## Files and privacy

- stage_private_checkpoints.py: fail-closed original provider progress, receipt, weights, adapter and SHA checks; creates only owner-private ZIP + private manifest on Windows.
- build_standalone.py: freezes original H419 source code SHA and creates H424 standalone Kaggle source, not a new train loop.
- kaggle/h424_checkpoint_eval_impl.py: actual free T4 model loads and original image-conditioned 240-case comparisons, no gradient steps.
- kaggle/kernel-metadata.json: PRIVATE original owner kernel with six original source datasets plus one private checkpoint dataset, no paid compute.
- tests/test_h424_checkpoint_evidence.py: 10 tests, including all SEVEN actual owner LoRA adapter preflight checks on Windows, negative provider progress and score policy tests.

Never upload any raw screenshots, Kaggle provider logs, individual original DEV prediction receipts, source archives, or original LoRA weights to public GitHub. All original raw weights live in Kaggle private and owner's private local staging only. This diagnostic does not finish Hala or approve medical/autonomous GUI use.
