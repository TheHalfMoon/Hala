# H25 HalaMind — prospective, non-executable DEV baseline

**Status:** CPU SOURCE + replay auditor only, not a Kaggle submission, completed model inference, qualified release, trained weights, visual grounding, or native Orcel integration. Branch stacked on Hala Draft PR #7. Historical H19/H22/H24 and H25 frozen files preserved, no other repository touched.

## Purpose

H24 unchanged Qwen3-VL-2B-Instruct V1 and V2 both measured **zero strictly accepted typed proposals** out of 20 reused synthetic DEV cases. Merely repeating prompt edits on those same 20 examples would lead to adaptive overfitting. H25 is a newly authored first-party prospective bilingual text-source candidate with 42 *public DEV* examples in three other application families (21 Arabic / 21 English). These are neither an unseen TEST nor evidence of real-world task performance.

The original H25 V1 inputs were label-adjacent. H25 projection V2 removes explicit reference-state booleans, predictable family/rank frame prefixes and generic one-word fake buttons. `h25/baseline.py` uses this **exact original-source-validated V2 model_input only**, never `source_ref` or private oracle fields in the model prompt.

## Fixed research candidate

- Public H25 frozen source SHA256: `b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b`.
- H25 V2 projection SHA256: `cceed2575f3f01995d489c811e2ee38f32272492fc2b533c0197ec0ba27c8ccd`.
- `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, planned FP16, no LoRA or quantization, no training; GPU not run by this change.
- Explicit seven `(tool,subgoal)` pairs, 42 per-case reference scores across stale frame, ready UI target, missing value, hostile page, destructive host command, persisted postcondition and contradictory signed evidence.
- Model must output strict `kind=next` JSON with exactly `kind,tool,subgoal`. All unrecognized, crossed or free-text tool/subgoal combinations fail closed. An invalid suggestion **never becomes an authorized action**.
- Preserve raw replies, rejected outputs, errors and a separate language/state breakdown; no output makes actual GUI actions. Orcel alone, in its own repository, controls native permission decisions and postcondition verification.
- Native Arabic text is Unicode verified at the authored source, but **not independently validated for semantics or RAQM/Type screenshot shaping**.
- `tools/verify_h25_dev.py` requires an **externally observed SHA256 for downloaded exact raw result bytes** and a pinned SHA40 Git source, checks fixed model/protocol/DEV-only identity, all 42 case labels/order/source, reported load and allocator values, and replays typed scoring from raw replies. This is CPU evidence-consistency checking, not physical-GPU attestation or independent semantic evaluator adjudication.

## Private Kaggle Saved Version, prepared but NOT submitted

- \`tools/build_h25_kaggle.py\` generates a **private** GPU+internet Kaggle Save & Run All notebook from an exact SHA40 review commit and a **new unused local output directory**.
- It does not allocate GPU, fetch weights, execute inference, or use Kaggle credentials. Notebook execution requires a real free Kaggle Batch session, unmodified reviewed source HEAD, strict versions and the remote-only model gate.
- Notebook code uses fail-closed explicit exceptions instead of optimizable \`assert\` guards. The exact-source pin is checked **before** model loading. It does not reuse H24 TEST or produce training gradients.
- Verify independent security and bilingual label-adjudication gates and four exact-head CI checks before deciding on **any** free Kaggle submission. No paid tier/upgrade or cloud API permission is inferred.
- An ephemeral saved-version run is not sufficient evidence: separately download raw 42-case results, hash them externally and recompute via \`tools/verify_h25_dev.py\`.
- Source-runner now requires consistent \`error\` classification in its own summary, rejecting a falsely labeled inference error on a present response or a missing bounded \`error_type\`. CPU oracle fixtures only prove test machinery, never model success.

After a reviewed exact SHA is published, *generation only*:

\`\`\`bash
python tools/build_h25_kaggle.py --source-commit FULL_REVIEWED_SHA40 --output NEW_LOCAL_NOTEBOOK_BUNDLE
\`\`\`

## Necessary next gate before any free research GPU use

1. Real fresh-context independent PStack/security review, bilingual expert semantic label audit, original-data rights assessment, and source-code provenance.
2. Confirm four exact-head GitHub CI jobs are passing on the signed/DCO publication, then generate and locally inspect a **separate private and immutable-source Kaggle Saved Version** with no paid plan, no billable resource upgrades, no credentials in stdout and no Windows model download.
3. Guard hardware and package versions on the actual remote free T4, record worker identity without claiming hardware attestation, save and download all 42 case outputs. Recompute independent raw-result SHA256, run the source-pinned CPU verifier, retain full raw evidence in Hala GitHub.
4. Report missing/invalid `kind`, malformed subgoals, wrong abstentions, unsafe `visual` suggestions and Arabic/English results by state. **Do not** tune on H25 DEV and claim it is unseen TEST.
5. Create genuinely unseen/protected multi-step families, legal training permission, a real HalaType Arabic visual/RAQM pipeline, and independently verified device task outcomes before any model improvement or release claim.
6. H23 real pending-CUDA-cancel, owned-child reap and restart remains a separately blocked lifecycle gate.

There are **no H25 GPU measurements**, no QwenGyre training, no model weights downloaded locally, and no merge/release or Orcel edits in this change. See `docs/QWENGYRE_RESEARCH_ADR.md` for the optional long-horizon research idea and the no-go on multi-node RL at this stage.
