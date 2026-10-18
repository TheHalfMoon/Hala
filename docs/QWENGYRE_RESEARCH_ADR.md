# Hala research ADR — QwenGyre applicability (2026-10-18)

**Decision:** research inspiration only. **Do not adopt or vendor QwenGyre code**; no Hala GPU or RL experiment is authorized from this paper alone. No changes to Orcel.

## Source

- Qwen announcement: https://x.com/Alibaba_Qwen/status/2109633085658808515
- Primary paper: Weiqi Wang et al., *QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents*, arXiv:2609.33848, submitted 2026-09-27. https://arxiv.org/abs/2609.33848
- QwenGyre is an **RL training infrastructure architecture**, not a released Hala-ready computer-use model or a replacement for our pinned Qwen3-VL-2B evaluator.
- At research time the paper did not link a verifiable public **QwenGyre framework** source-code repository, license or machine-portable implementation. Do not assume Apache/MIT permissions merely because other Qwen repositories use them. Do not copy an unpublished implementation.

## Paper claims, with evidence limits

- Elastic GPU scheduler reallocates capacity between sampled long-running agent rollouts and training without interrupting harness execution. This targets expensive multi-node GPU idle time.
- Trajectory processor reconstructs branching execution trees (retries, delegated sub-tasks, context compaction), scores partial-progress snapshots and caps overlapping training samples.
- Reported flagship Qwen 3.8 2.4T experiment on NL2RepoBench: success about **52.5% to 58.5% in 48 RL updates** (an absolute increase of about six percentage points); at matched GPU budget, **75.42h vs 134.55h** Async baseline, 1.78x elapsed-time speedup. Across reported cases up to **1.85x** relative to Colocate.
- These are **author results at multi-node scale**, not independent Hala replication; hardware/cost/energy claims do not transfer to one free Tesla T4, and speedups do not imply a reduction in minimal GPU requirements.
- Paper/results not proof of real Arabic GUI task safety, valid typed proposals, licensed code reuse, or cheaper commercial inference.

## Adoptable ideas for Hala (no new RL cluster)

1. **Trajectory provenance before rewards.** Keep source model SHA, split, locale, device/frame identity, typed decision, untrusted-page provenance, policy outcome and actual independently observed postconditions. No fabricated task success from the model's own text.
2. **Shared-prefix deduplication (future).** On *authorized original synthetic* episode trees, compute parent/child hashes; retain distinct causal branches while bounding near-duplicate samples. Splits and users must remain isolated; do not leak private real-device telemetry.
3. **Partial-work credit (future).** A timed-out trajectory is not an automatic success. A separate evaluator may score independently verified intermediate states with explicit stop/error reasons, uncertainty and audit hash.
4. **Durable free-worker recovery now.** Persist small, exact-source, per-case results promptly; verify exact byte hashes after retrieving a saved version. Never pretend ephemeral Kaggle GPU state survived across sessions.
5. **Single-T4 affordability first.** Fix HalaMind typed proposal validity, bilingual task routing, and real end-to-end action/verification success before RL infrastructure or elasticity. No multi-node scheduler, Megatron/SGLang cluster, 700k-token rollouts or paid cloud subscriptions.

## Explicit no-go gates

- No QwenGyre model swap (it is a training framework); Hala model pins unchanged.
- No copying unpublished framework code or inferring a license from the research paper.
- No training on frozen H22 TEST, H24 TEST or test-contaminated trajectories; H25 public DEV is not unseen TEST.
- No GPU spend, cloud/API bills, sustained hosted inference, or local large-weight download.
- No claims of 1.78x Hala speedup, 6% Hala accuracy gain, energy savings or production scale.
- No autonomous policy execution, Orcel source change or weakened native authority.

**Follow-up research gate:** recheck an official QwenGyre code release and license if announced; independently evaluate a CPU-only prefix/dedup/progress simulator with frozen synthetic data and separate brand-new task-family holdout. Introduce GPU-based agent RL only after independent reliable HalaMind/Type baseline and founder approval.
