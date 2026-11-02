# H350 — Historical Mind/Type trainer cannot bypass official H199 admission

The executable `training/scripts/train_development.py::train` is a separate remote Colab PEFT/LoRA entrypoint from the native GUI trainer guarded in H346. Previously it accepted a frozen development bundle and remote training opt-in without requiring independently approved H199 admission of the exact manifest bytes. Frozen source integrity is not a screenshot ML training grant.

H350 checks a bounded `manifest.json`, rejects bundle/manifest symlinks and mismatched external digest, and requires its exact bytes to be admitted in `h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS` **before** remote setup, baseline, model, processor, GPU, optimizer or output creation. Production H199 is EMPTY, so this executable path fails closed. Historical `preflight`, original evidence and research outputs remain unchanged.

Six invented-only CPU tests cover self-certified rights denial, test-scoped allowlist byte binding, digest drift, symlinks, oversize and actual `train()` refusal before side effects. A mocked test registry does not approve any real data.

This is one concrete alternate entrypoint, not an audit of every historical kernel. No new training, model inference, legal approval, blind OOD, learned gain, release, merge or deployment. Independent source title/screenshot ML rights and separately custodied family-disjoint DEV/OOD remain blockers.
