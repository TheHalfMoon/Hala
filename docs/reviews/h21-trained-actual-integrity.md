# Actual trained adapter integrity review

PASS for artifact/record integrity; no findings. Verified the externally anchored manifest `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d` through the reviewed archive verifier and independently checked the actual parameter payload without Torch or other ML imports.

- Exactly four exported bundle files; no optimizer pickle or base weights. Model is 3,227,648 bytes, SHA256 `7eb2f17df11151e5b77a5c715c59f56c854ca7b0d0da3ee591f394b501826fa7`.
- All 112 exact language q/v rank4 tensors have correct shapes, contiguous complete offsets and F32 dtype. Independently decoded all **802,816 scalar values** with `struct`; every value is finite. Config/file hashes match original retained bytes, pinned base revision, alpha 8, dropout 0 and PEFT producer target regex. Config revision is the pinned revision; no config rewriting occurred in this review.
- Full source/data/DEV-baseline identity matches the externally supplied pretraining identity. Published training source `5b6eb4bf29acbaf21aa199f113f78af4f268b3b7` contains script SHA256 `17f0270cb721d343aecb5dcab1d35b3c88fcdf3735e3cccf2e180189bd7790f1`.
- Exactly 116 ordered step metrics match the frozen authored Mind/Type train row IDs, excluding Reliance and held-out rows. Schedule remains one epoch, seed 42, rank4/alpha8, AdamW 5e-5, FP16, initial scale 1 and gradient clipping 1.
- Every recorded loss, norm and step duration is finite; every norm is positive and every step records 112 changed tensors. Final aggregate records 112 changed tensors. Loss range is 0.000353–2.090938; gradient norm range is 0.023051–16.109512. The logged norm is the pre-clipping norm returned by the clipping operation; values above 1 do not contradict the configured clipping threshold.

All four bundle files and the 68,480-byte training result match Git objects at retained archive commit `274024e14699599a51fca2c948af40819532e18c`. Training result SHA256 is `905368e88c1734310142d0f69819cdca1df8ae0f217777418f4dc9f48f0b7db0`. Historical status `trained_pending_durable_archive` is preserved as originally emitted; it was not rewritten after later retention. Loss and finite weights provide no evidence of task improvement or Arabic/native quality.

Trainer wall time is recorded as 116.09 seconds; peak allocated CUDA is 5,773,505,536 bytes, peak reserved 13,035,896,832 bytes, and maximum sampled device usage 13,238,140,928 bytes. Reserved/device usage differs from active tensor allocation; these figures do not establish full-session or production cost.

Limits: discarded gradient tensors and before-step parameters are unavailable for independent numerical replay, so per-step updates/gradient finiteness are recorded evidence backed by the pinned source guards; parameter payload finiteness is directly verified. Local signature verification could not run because `gpg.ssh.allowedSignersFile` is not configured; this review does not independently attest the signature or contradict the coordinating operator's signed-retention verification. Fresh public retrieval, real PEFT reload and frozen independent-case evaluation remain separate gates.

Audit script: `work/h21_trained_actual_integrity.py`; result: `work/h21-trained-actual-integrity.json`; exit 0. No repository source changes, new training, ML installs/imports or GPU work by reviewer.
