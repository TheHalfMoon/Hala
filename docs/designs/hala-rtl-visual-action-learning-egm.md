# H330: Hala RTL visual-action learning — EGM design draft

**Status:** PROPOSED — NOT REVIEWED, NOT APPROVED, NOT IMPLEMENTATION-READY.
**Design process:** [Google Elephant–Goldfish Model (EGM)](https://github.com/google/elephant-goldfish-model) at upstream commit `403d8881d62022490637d740657b33f01e321ea8` (2026-10-30), Apache-2.0. This is an original Hala project design that follows EGM's feature specification standard; it is **not** a claim that upstream EGM was installed, that a Goldfish subagent reviewed this text, or that Google endorsed Hala.
**Decision owner:** Hala founder. **Next state:** isolated Goldfish review and owner resolution of open design choices before any new model training.

## 1. Problem and observed evidence

Hala is intended to execute screenshot-conditioned Arabic/English GUI tasks with native, safe, two-step actions. Successful CI exercises source-owned browser oracles, **not** model competence. Existing Hala evidence supports a specific failure to investigate:

| Source | Actual observation | What it does not prove |
| --- | --- | --- |
| H291/H292 | Genuine paired original BASE 6/8, adapted LoRA 6/8 | No measured learned improvement |
| H300 | Genuine pretrained Qwen FieldPort 12/12 on known-source exposed tasks | Not blind generalization |
| H324 | Owner-private FREE T4 pretrained Qwen BeaconBoard EN 1/1 full task; AR RTL 0/1 (wrong focus/state after first click), only 3 total model generations | N=1 per language, not evidence of general Arabic failure |
| H325/H327 | 16 first-party fictional RailKit source scenes, 32 staged source-gold TRAIN candidate messages, zero model generations | NOT official TRAIN admitted |
| H326 | Known-correct source points pass 12/12 native H273 normalized conversion; six mirrored RTL negatives fail as intended | Does not measure learned localization |
| H328 | 32 RailKit and 24 exposed BeaconBoard decoded RGB frames with zero exact collisions | Does not prove semantic disjointness or unseen OOD |
| H329 | Exact RailKit Git introduction + unchanged Apache-2.0 LICENSE and no external HTML assets verified; introducing commit unsigned | Not independent copyright/title, screenshot license, or training permission |

The primary gap is **no legally admitted training data plus no independent source-disjoint blind test or paired learned improvement**. H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is empty. H173 scored OOD is SEALED; H212 barred; H250/H251 rights-quarantined; H293/H294 private retrieval prohibited. No proposal may reopen these.

## 2. Goals and exclusions

**Goals:** (a) establish an explicit, reviewable provenance-and-admission decision per first-party training cohort; (b) isolate whether Qwen's RTL error occurs in target identification versus coordinate conversion, focus, or action parsing; (c) preregister a low-cost, paired baseline-versus-adapted study with equal model and sample budgets; (d) prevent leakage from exposed source families into a genuinely new blind OOD cohort; (e) keep unsafe or unsupported branches inert.

**Not goals:** declaring improved accuracy from mocked fixtures; training on H325 candidates before evidence-backed authorization; downloading private media or model responses forbidden by H293/H294; changing H273 coordinate semantics without an independent counterexample; using H173/H212 historical tests; spending money or claiming release-readiness from a single example; re-implementing Google's EGM library inside Hala.

## 3. Roles and custody boundaries

`h325/railkit.html` is author-claimed first-party, synthetic and code-generated. H329 established a specific origin commit, but **Git authorship metadata alone is not a legal rights grant**. A genuinely independent reviewer must inspect source custody, document any donors/derivatives, confirm commercial/noncommercial training and screenshot-output scope and sign an explicit permission record. Until then H199 remains unchanged, any generated visual/action pair stays ephemeral, and no training run is authorized.

Never send screenshots, prompt/gold or model outputs into public PR comments, CI logs, GitHub Actions artifacts, plugin prompts, Goldfish packets, or generic file upload. EGM reviewers get **this design and linked public source paths only**; not Kaggle private receipts, sealed OOD or H325 candidate contents.

The Elephant has the full historical engineering context and writes designs. Goldfish reviewers must be fresh-context, *read-only* agents with no parent transcript; their actual native launch/result traces must show isolation, document/standard reads and completed review. A parent-authored critique, a single-thread simulation, or review by this chat **must not** be marked as a Goldfish review. Missing true fresh-context support is `REVIEW_BLOCKED`, never `PASS`.

## 4. Candidate architecture (not yet approved)

```text
Source Git/ownership evidence [H329] --independent legal decision--> admitted TRAIN cohort [H199]
       |                                                  |
       | prior exposed families excluded                  v
       |                                      private, zero-cost staged adapter TRAIN
       |                                      model revision and data pins fixed
       |                                                  |
fresh *unexposed* source family + preregistered owner-only blind split
       |                                                  |
       +-----> frozen paired baseline / adapted model-time evaluation <---+
                         |
           safe aggregate counts only; fail-closed release gate
```

The training/source validator must reject any candidate if the input source lacks a pinned Git blob, independent rights and generated-screenshot scope, verified no overlap with evaluated families, or an official reviewed H199 admission. A weak Git timestamp, unsigned commit, AI-generated source claim or an EGM review alone must never satisfy rights. Keep independent development examples separate from the final hidden blind OOD; after one exploratory run those dev examples are exposed.

Model evaluation is ordered per case: fixed screenshot → model-proposed native action → parser and coordinate validation → actual native browser dispatch → *source-only* state/focus check → second screenshot when applicable → model next action. Never reveal DOM/target coordinates to policy. Capture typed `abstain`, malformed actions, incorrect targets, wrong focus, and early stopping separately, not just a single accuracy label. Only aggregated categorical results may leave owner-private evaluation.

## 5. Alternatives considered and trade-offs

| Alternative | Benefit | Why not select blindly |
| --- | --- | --- |
| **A. Tighten Qwen prompt and screenshot target representation, no weight updates** | Low complexity, zero TRAIN admission needed; tests whether model inference can use existing vision better | Prompt tuning can overfit exposed examples and does not show learned LoRA gain; requires separate DEV split |
| **B. LoRA on independently cleared first-party RailKit plus other cleared source families** | Directly targets weights and bilingual click/focus behavior | Only 32 candidate actions today, likely insufficient source diversity; rights admission and free quota not complete; high overfit risk |
| **C. Hybrid: deterministic UI geometry postprocessor with explicit local evidence, plus frozen Qwen** | Improves native action robustness if a measurable conversion defect exists | H326 ruled out simple RTL x-axis reversal at correct button centers; cannot repair incorrect visual target choice and may conceal failure |

**Design decision pending:** choose A/B/C only after isolated review and independently verified prerequisite evidence. Do not silently commit to a training plan or treat this document as founder approval.

## 6. Preregistered measurements and acceptance

Pair **the same frozen image, task, sampling configuration, action schema and native browser policy** across pretrained and adapted models with no later cherry-picking. Record model repository and exact revisions, LoRA checkpoint hash, rights/admission commit, family split manifest identifier, random seed, cost/quota eligibility, and versioned evaluation grammar before dispatch. Do not measure using training source scenes as if blind.

Report total tasks and by-language exact task success, first-click correct target, correct input focus, typing success, bounded abstentions, invalid-format rate, timeout and missing-report rate. Count every scheduled task including refusals/errors in denominators. A sample is `task_success` only if source-owned final state passes **both** steps; a correct click alone is not full success.

**No numerical performance target is claimed from H324 N=2.** Before computing a credible success threshold, the reviewer must justify an independent sample size and uncertainty method (paired case-level CI, not just point percentages), source-family diversity and false-positive risk. If there are insufficient valid independent cases, label `INCONCLUSIVE` instead of `GAIN`. A zero/zero or one/two denominator is never release evidence. Train source and exposed BeaconBoard/FieldPort may be used only as public regression/debug sets, not fresh blind OOD.

A release requires all of: legal TRAIN admission; independent held-out evaluation; preregistered unchanged analysis; meaningful BASE vs ADAPTED win with uncertainty; bounded malformed/unsafe actions; security and privacy sign-off; founder release approval. Missing any = `NO_GO`.

## 7. Failure/abuse modes and mitigations

1. **Data rights laundering:** author Git claim + Apache license mistaken for independent media/training rights. Mitigate via explicit outside legal/source reviewer and immutable approval, H199 remains empty until accepted.
2. **Information leakage into Goldfish:** hidden gold or parent history passed with review packet. Mitigate with source-only reading list and fresh-context trace check; no logs/receipts attached.
3. **Overfitting to invented 16 scenes:** apparent near-perfect result from source-owned oracle. Mitigate new never-exposed independent family and narrow adaptation budget only after adequate train diversity.
4. **RTL target localization confounding:** confusing model language comprehension with wrong pixel conversion. Mitigate separate first-click model prediction and source-native coordinate oracle, no DOM passed to model.
5. **GPU/CI duplication and cost:** stacked PR workflows plus repeated Kaggle pushes consume quota. Mitigate one frozen job at a time, owner-private free-only eligibility, stop on 400/blocked, no retries without evidence.
6. **Evaluation privacy breach:** JSON, screenshot, prompt, gold or model predictions accidentally exposed in review reports. Mitigate strict aggregate-only receipt allowlist and local cleanup verification.
7. **Unverified EGM subagents:** installing skills mistaken for running truly fresh reviewers. Mitigate explicit `NOT_RUN`/ `BLOCKED` stages until native isolation and completion proof exists.

## 8. Implementation phases (no implementation permission implied)

**Stage 0 — EGM design review:** deliver this source-linked draft and alternatives to a fresh Goldfish with upstream standards `feature-spec-standard.md`, `implementation-readiness-standard.md`, and personas `agents/zoom_out.md`, `agents/comprehension.md`, `agents/design_critic.md`, `agents/implementation_readiness.md`. Persist results and unresolved issues in a separate project design review PR, without private sample content. If fresh-context agents are unavailable, `REVIEW_BLOCKED`.

**Stage 1 — Rights decision:** separately verify author/custody, source HTML and generated visual rights, donor absence and permitted use; collect independent approval before touching H199 TRAIN admission; if denied, stop.

**Stage 2 — Method preregistration:** produce a source-family split specification without accessing hidden blind gold; specify target counts, failure categories, baseline and adapter pinned revisions, inference conditions, complete task denominators, uncertainty and abort conditions; reject source overlap and exposed reuse.

**Stage 3 — Candidate integration:** if cleared, design and implement train-only rendering and batching, no external persistent candidate assets. Keep source/oracle separable from policy and require atomic failure. Add synthetic and browser regression tests.

**Stage 4 — Model-time study:** only after quotas, privacy and scientific gates are verified, perform one approved FREE, owner-private paired run; safe aggregate-only exports. Stop on blocked access and do not retry in ways that evade restrictions.

**Stage 5 — Evidence/decision:** source-backed analysis with honest uncertainty; mark `NO_GO` when no independently measured improvement; publish no weights or release without founder authorization.

## 9. Design questions for true Goldfish (currently unanswered)

- Is 32 train candidate actions from 16 closely related scenes scientifically inadequate even with an adapter? What additional independent source families are needed?
- Is an explicit independent legal reviewer required for first-party HTML rendering derivatives, and what exact evidence would suffice?
- Does the H324 Arabic failure support a model-localization hypothesis or merely a one-off state/focus anomaly?
- How can an evaluator verify that blind tasks were not exposed in prior H300/H324 experimentation?
- Can native fresh-context reviewer isolation be verified by the selected EGM harness on the actual owner environment without exposing private material or incurring paid inference?
- How should task-level paired uncertainty and language subgroup reporting be preregistered to avoid posthoc claims?

## 10. Pinned EGM sources and references

- EGM repo/commit: https://github.com/google/elephant-goldfish-model/tree/403d8881d62022490637d740657b33f01e321ea8
- Spec standard: `resources/document_standards/feature-spec-standard.md` at that commit.
- Readiness standard: `resources/document_standards/implementation-readiness-standard.md`.
- Codex guide: `docs/codex.md`. Goldfish requires genuine fresh native subagents; **not available in this ChatGPT exchange**.
- Hala source read list: `h199/native_action_sft.py`, `h291/README.md` (if present), `h292/README.md`, `h319/beaconboard_dynamic_research.py`, `h324/verify_private_beaconboard_receipt.py` (code only), `h325/two_step_train_candidates.py`, `h326/rtl_relative_geometry_audit.py`, `h328/original_family_visual_split.py`, `h329/railkit_source_provenance.py`, `AGENTS.md`.
- Current artifact is a **design proposal**; no Goldfish reviewer has been dispatched or marked passed and no architectural approval has occurred.
