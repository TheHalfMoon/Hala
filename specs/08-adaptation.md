# H8 candidate adaptation and paired qualification

Depends on: real H5 deficit and H7 content-bound train artifact. Surface: training, tests/test_training_integrity.py.

Acceptance: original train-only PNG/prompt/logical-action exports and media hashes; Arabic images excluded until shaping qualifies; strict prompt/vision completion masks; base/vision frozen; finite actual gradients and changed LoRA parameters; accumulation/checkpointing; interruption-safe paired adapter/optimizer/scaler/RNG snapshots; exact config/data/artifact digests; completed resume finalization; independent heldout base-vs-adapter evaluation with uncertainties. Compare shared and specialized adapters only for a measured deficit.

Hardstop: candidate helper tests do not establish actual model training. No fictitious adapter, resume equivalence or performance improvement. QLoRA remains unimplemented until a supported quantized training path is exercised.
