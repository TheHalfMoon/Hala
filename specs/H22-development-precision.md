# H22 — Representative closed-loop precision and process cost

Status: source CPU-tested; independent gates and actual GPU qualification pending. Review: full.
Depends on: frozen H19, H21 verified base/adapter runtime. Surface: optional evaluation precision loader, observational cost sampler, pairing report, tests and prospective configuration/freeze. No training, serving, Aladdin or historical H14/H17 changes.

Compare fp16, int8 and nf4 on the exact frozen H19 bilingual closed-loop dev or test allocation, same stored order, prompts, parser, simulator and independent final-state/semantic verifier. Use Qwen revision 89644892e4d85e24eaac8bacfd4f463576704203, seed42, SDPA,128 output/4096 input tokens,20 second generation and existing episode/outer budgets. Standalone Mind diagnostics retain their existing allocation. Precision profiles are separate from frozen FP16 reference SETTINGS. Initial base versus retained adapter choice remains pending actual H21 quality; do not adopt or retrain.

Use explicit provider admission inherited from H21f, including genuine Kaggle context/inherited image tag; never spoof environment admission. No local weights, cache escape or CPU offload. Preserve existing exact package pins; bitsandbytes0.50.2 is the historically executed H17 version, requiring fresh actual compatibility qualification. Same externally verified complete retained adapter identity when selected. Publish prospective config/source hashes and require external freeze digest before GPU execution; do not self-qualify from CPU.

Acceptance criteria:

- [x] Frozen SETTINGS/QwenInfer/training guard/default FP16 behavior preserved; precision preimport admission/settings/source/adapter guards tested red then green.
- [x] Shared runtime loads exact three profiles, enforces full model/vision/quant-state CUDA residency and records exact model/config/adapter/task identities.
- [x] External bounded fail-open sampler source records process RSS/CPU time and best-effort device used/free/utilization/power; model storage, unique physical cache bytes, load/per-call latency and allocator/reserved/device distinctions remain explicit. Actual observations pending.
- [x] Pair report rejects duplicate/missing tasks or changed source/profile/prompt/model/adapter identities; independent final-state and semantic metrics retain en/ar groups. No quality/adoption claim from CPU or historical64 fixtures.
- [x] Scoped red/green and full CPU gate, independent reviews/Jev/Graft and signed source before operator external freeze and actual execution.

Sampled maxima are observations, never continuous hardware peaks. Missing telemetry is null with an error label and must not remove inference evidence. Full cost refers to the evaluation process/device observations, not price or production service qualification. Actual GPU runs and artifact publication are operator work after review.

Source handoff, exact hashes and operational limits: [H22_PRECISION](../docs/H22_PRECISION.md). Seven initial scoped criteria tests ran red before implementation; the reviewed source gate233 tests passed with one skip. Independent combined review then requested correct logical/physical GPU attribution; security review noted cache traversal bounds, and the combined reviewer noted unsupported-field labels and cadence wording. Three repair criteria tests ran red, then scoped38 tests passed, including15 H22 tests. The final repaired source gate is recorded in the builder report. No independent judges or Jev/Graft were run by the builder; no commit was made.

Final delta closes the remaining admission traversal bound and binds the resolved physical GPU UUID into pair identity. Admission rejects a scan budget overrun before dependency/model imports; unknown UUID pairs explicitly leave physical-device cost unqualified. Both regression surfaces ran red, then the scoped39 tests passed with16 H22 tests. Final full gate and refreshed freeze identity are in the builder report; earlier Jev/Graft hashes require conductor requalification.
