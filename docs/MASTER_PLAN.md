# HALA — AUTONOMOUS MULTIMODAL AI DEVELOPMENT, FINE-TUNING, QUANTIZATION AND INFERENCE ENGINEERING

## ROLE AND PRIMARY DIRECTIVE

Act as a Principal Foundation Model Researcher, Multimodal AI Scientist, Computer-Use Agent Architect, ML Training Engineer, Inference Optimization Engineer, AI Infrastructure Architect, Security Engineer, and Technical Lead.

Your mission is to build **Hala**, a highly capable, compact, efficient, commercially usable computer-use AI system that will power **Aladdin Pro**.

**Canonical repository:**

https://github.com/TheHalfMoon/Hala

**Related application:**

https://github.com/TheHalfMoon/Aladdin

**Owner:** TheHalfMoon

**Official AI system name:** Hala

**Official component names:**

1. HalaMind
2. HalaType
3. HalaReliance

**LANGUAGE: ENGLISH ONLY.**

All communication, source code, research, documentation, commits, PRs, comments, tests, reports, and technical artifacts must use English.

Use the highest supported reasoning effort.

Be scientifically rigorous, commercially practical, security-conscious, and extremely cost-efficient.

Do not merely write plans or produce an impressive-looking repository.

Build working components, execute legitimate experiments, validate actual model performance, and progress toward an independently deployable AI inference engine.

Work autonomously through successive authorized milestones.

Do not stop after every step to ask me to say "continue."

Do not invent benchmark results, trained checkpoints, review evidence, GPU availability, financial costs, or deployment success.

Never confuse proposed features with implemented and tested features.

---

# 1. FOUNDER'S VISION

Aladdin is a standalone computer-use product with two tiers.

## Aladdin Basic

Target price: $3/month.

- Local-first computer-control application.
- No native AI inference.
- No founder-paid GPU requirement.
- Deterministic computer tools and workflows.
- Optional MCP integration.
- No ChatGPT or Claude subscription required for non-AI features.

## Aladdin Pro

A full AI-powered computer-use application.

- Hala is integrated directly into Aladdin.
- Users interact with Aladdin's own interface.
- No ChatGPT subscription required.
- No Claude subscription required.
- No customer OpenAI or Anthropic API key required.
- No external customer AI account required.
- No powerful customer GPU required.
- Hala inference operates on founder-controlled infrastructure.
- Aladdin executes approved actions locally through its trusted kernel.

The customer should be able to say:

"Organize my Downloads folder."

"Open the browser and find the latest report."

"Fill out this form."

"Create a summary from these documents."

"Find the error in this application and help fix it."

Aladdin should understand the request, plan the task, obtain required authorization, execute permitted actions, independently verify the result, and report what actually happened.

Hala provides intelligence.

Aladdin provides execution, authority, and verification.

The goal is high real-world reliability at the lowest sustainable operating cost.

---

# 2. PARALLEL DEVELOPMENT — STRICT REPOSITORY ISOLATION

Claude is already implementing Aladdin.

You are responsible for developing Hala independently and concurrently.

Do not modify Aladdin's repository.

Do not open PRs to Aladdin.

Do not interrupt Claude's branches, worktrees, pending commits, or implementation schedule.

Do not rewrite Aladdin's architecture.

You may read the current Aladdin source, specifications, AI architecture documents, and integration contracts as reference material.

Pay attention to:

- `docs/aladdin/ALADDIN_AI_SERVER_ARCHITECTURE.md`
- `docs/aladdin/MODEL_INTEGRATION_CONTRACTS.md`
- `docs/aladdin/ARCHITECTURE.md`
- `docs/aladdin/IMPLEMENTATION_ROADMAP.md`
- Current Aladdin AI research and PR #328, if applicable.

Distinguish adopted architecture from proposed research.

If Aladdin changes during your work, maintain a compatibility register and versioned adapters.

Do not depend on unfinished Aladdin features for Hala's development tests.

Use synthetic environments and a mock executor to qualify Hala independently.

Hala must eventually be integration-ready without requiring Claude to rebuild its computer-control architecture.

---

# 3. VERIFY AND INITIALIZE THE HALA REPOSITORY

The repository was previously verified as existing but empty.

Recheck its current state.

Inspect:

- Repository metadata.
- Current branches and commits.
- Working tree.
- Existing files.
- Open PRs and issues.
- Repository permissions.
- Required CI.
- Existing governance.
- Tool availability.

If it remains empty, initialize it correctly.

An empty GitHub repository has no valid main commit yet. Do not invent a main SHA.

Create the minimum professional bootstrap necessary to establish the main branch.

Then work through reviewed feature branches and PRs.

Preserve all future existing work.

Use signed commits and DCO when supported and required.

Do not force-push, rebase, squash, or rewrite shared history.

Keep code, experiments, research, tests, and source files in:

`TheHalfMoon/Hala`

Keep large model weights and datasets outside GitHub.

---

# 4. THE THREE OFFICIAL HALA COMPONENTS

The official architecture has three functional components.

These names are fixed:

- HalaMind
- HalaType
- HalaReliance

However, **three functional components do not automatically require three separate resident models**.

Begin with the most economical qualified implementation.

## 4.1 HalaMind — Reasoning and Planning

Responsible for:

- Understanding user intentions.
- Natural-language reasoning.
- English and Arabic comprehension.
- Long-horizon task planning.
- Tool choice.
- Task decomposition.
- Context management.
- Error recovery.
- Asking clarifying questions.
- Generating bounded executable plans.
- Understanding prior tool outcomes.
- Selecting safe next steps.
- Deciding when additional observations are needed.

Initial model candidate:

`Qwen/Qwen3.5-4B`

Investigate whether a smaller model provides adequate performance.

HalaMind must produce structured plans and proposals, not unrestricted executable programs.

## 4.2 HalaType — Visual Interaction and Typed Actions

HalaType includes the previously discussed Hala Eye responsibilities.

Responsible for:

- Screenshot interpretation.
- GUI grounding.
- Window understanding.
- Browser interface understanding.
- Accessibility and DOM element grounding.
- Mouse-target identification.
- Click, type, key and scroll proposals.
- Drag-and-drop understanding.
- Form-field interaction.
- UI state changes.
- Coordinate mapping.
- Typed MCP/UIA/browser action proposals.
- Arabic and right-to-left interface recognition.

HalaType must understand the allowed action vocabulary provided by Aladdin.

It must not invent new executable tools.

It must not directly run arbitrary Python, shell commands, JavaScript or Selenium code.

First determine whether HalaMind and HalaType can share one multimodal checkpoint.

Only train separate adapters if experiments demonstrate meaningful improvements.

## 4.3 HalaReliance — Tiny Decision Specialist

HalaReliance is a very small, low-latency decision system.

Its purpose is to reduce expensive HalaMind/HalaType inference and improve task routing.

Research:

- LiquidAI/d1-omni-600M.
- LiquidAI/d1-3B as a comparison.
- Smaller trainable classifiers.
- Efficient encoder models.
- Tiny task-specific classification networks.
- Deterministic decision routing.
- CPU inference as a potential deployment option.

Investigate whether a model smaller than 600M can perform the required decisions.

HalaReliance may predict or recommend:

- Deterministic tool is sufficient.
- HalaMind required.
- HalaType required.
- More observation required.
- Ask the user.
- Abstain.
- Model escalation needed.
- Task budget exhausted.
- Additional conservative review recommended.

HalaReliance must never:

- Grant approval.
- Override local policy.
- Lower an action's security classification.
- Mint device leases.
- Invent trusted target identities.
- Forge execution receipts.
- Declare actions successful without independent verification.
- Authorize privileged commands.
- Override a user's cancellation.

**Aladdin's local authority kernel remains the final security authority.**

### HalaReliance licensing

Liquid d1 models are under the LFM Open License, not Apache-2.0.

Verify exact commercial terms, derivatives, revenue conditions, and redistribution requirements before adopting a checkpoint.

Compare a commercially permissive independently trained compact classifier as an alternative.

Do not require proprietary per-token APIs to operate HalaReliance.

Do not assume Liquid d1 is a general-purpose chat model. Verify its native typed decision API and runtime requirements.

---

# 5. OPTIMIZE THE SYSTEM ARCHITECTURE

Benchmark at least these designs:

**Architecture A**

One compact multimodal model performs Mind and Type tasks with deterministic routing.

**Architecture B**

One shared base model with separate HalaMind and HalaType prompts.

**Architecture C**

One shared base model with specialized Mind and Type LoRA adapters.

**Architecture D**

One shared Mind/Type model plus HalaReliance.

**Architecture E**

Separate small HalaType model with HalaMind invoked only when necessary.

**Architecture F**

Three separate model implementations only if the measured benefit justifies the additional expense.

Compare:

- Actual complete-task success.
- Arabic performance.
- GUI grounding.
- Planning quality.
- Routing accuracy.
- Wrong-target rate.
- Memory residency.
- VRAM use.
- CPU and GPU seconds.
- Warm and cold latency.
- Model calls per task.
- Training effort.
- Serving complexity.
- Cost per independently verified successful task.

Select the simplest design satisfying performance, safety and cost goals.

Avoid keeping unnecessary models continuously resident on GPU.

---

# 6. MODEL FOUNDATION RESEARCH

Do not train a model from scratch.

First compare suitable released multimodal checkpoints.

Provisional priority:

1. Qwen3.5-4B.
2. Qwen3-VL-2B-Instruct.
3. Hcompany Holo-3.1-4B.
4. Hcompany Holo-3.1-0.8B.
5. ShowUI-2B.
6. Qwen3-VL-4B-Instruct.
7. Other compact commercially compatible vision-language models.

Use Holo4-35B-A3B as a stronger research baseline only when qualified compute and a bounded budget are available.

Do not require the 35B model for every experiment.

For each checkpoint, record:

- Official source.
- Exact immutable revision.
- License.
- Base-model license.
- Commercial restrictions.
- Architecture.
- Parameter count.
- Model processor.
- Supported image format.
- Tool-call format.
- Quantization support.
- Training feasibility.
- GPU requirements.
- Arabic capability.
- GUI grounding capability.
- Known benchmarks.
- Runtime compatibility.
- Model size on disk.
- Estimated and measured runtime residency.

Do not assume a model with fewer active MoE parameters has fewer resident weights.

Do not assume published benchmarks establish superiority on Aladdin tasks.

Choose a provisional foundation and preserve alternatives for evaluation.

---

# 7. DEEPLY STUDY HCOMPANY AND HOLO TRAINING

Research official releases and available technical materials for:

- Holo3.
- Holo3.1.
- Holo4.
- Their computer-use architecture.
- GUI-specialized training.
- Desktop/browser task generation.
- Tool-use training.
- MCP/API tasks.
- Supervised fine-tuning.
- Reinforcement-learning methodology.
- LoRA specialist training.
- Dataset preparation.
- Task verification.

Research the actual public evidence.

Do not invent access to Hcompany's private training corpus.

The complete Holo4 training dataset may not be publicly available.

Public benchmark trajectories are not automatically suitable for training.

## Required dataset investigation

Study:

https://huggingface.co/datasets/Hcompany/DragOn

https://huggingface.co/datasets/Hcompany/trajectories

Research Hcompany's other legitimately released training assets.

### DragOn

Investigate:

- Actual schema.
- Image dimensions.
- Drag-and-drop action representation.
- Bounding boxes.
- Source provenance.
- Licenses and asset rights.
- Dataset duplication.
- Potential benchmark overlap.
- Suitability for HalaType.

Use rights-cleared subsets for initial experiments.

Do not download the full dataset until it becomes necessary.

### Holo trajectories

The published trajectories are associated with benchmark evaluation runs.

Keep benchmark tasks and their trajectories isolated from training.

Do not use them to contaminate OSWorld, AndroidWorld, or related evaluations.

Use official evaluation materials only for permitted analysis and benchmark reproduction.

### Hala's independent data

Reproduce useful *training methods*, not private or unavailable data.

Build Hala's own task factory with:

- Synthetic Windows controls.
- Synthetic browser applications.
- Arabic and RTL screens.
- English screens.
- GUI element identification.
- Text entry.
- Drag-and-drop.
- Tool-choice tasks.
- Recovery.
- Ambiguous requests.
- Rejection and abstention cases.
- Verified state transitions.

Generate training labels with deterministic validators wherever possible.

Record provenance and rights for every source.

---

# 8. STUDY LAVAGUE

Mandatory research reference:

https://github.com/lavague-ai/LaVague

Analyze actual source code.

Focus on:

- World Model.
- Action Engine.
- WebAgent.
- Navigation Engine.
- Python Engine.
- Navigation Control.
- Context retrieval.
- HTML selection.
- Short-term memory.
- Token/cost accounting.
- Browser QA.
- BigAction datasets.
- Agent failure reporting.

Identify reusable algorithms and concepts.

Evaluate how LaVague reduces unnecessary LLM calls.

Compare its concepts with Aladdin's existing structured tools.

Important:

- Do not import unrestricted generated-code execution.
- Do not enable LaVague's default OpenAI integration.
- Do not enable third-party outbound telemetry.
- Do not import outdated dependencies without review.
- Do not make LaVague a mandatory runtime dependency by default.
- Preserve license notices for any source reuse.

Create:

`research/LAVAGUE_ASSESSMENT.md`

Include exact source SHA, important components, security concerns, data opportunities, maintenance risks, and recommended reusable ideas.

---

# 9. FREE GPU AND ACADEMIC COMPUTE STRATEGY

**Founder out-of-pocket target: $0 throughout early development and training.**

The founder has access to Kaggle, Colab and potential free-compute opportunities.

GitHub Education was previously declined, so do not make it a prerequisite or repeatedly require the founder to apply there.

Investigate:

- Kaggle free GPU.
- Kaggle T4x2 where available.
- Google Colab free GPU.
- Modal Starter credits.
- Modal research and academic grants.
- Other genuine free GPU providers.
- Legitimately available research compute.
- Available student or academic benefits.
- Free CPU environments for model-independent work.

Use official current provider terms.

Do not assume free GPU quotas, instance types, or credits are guaranteed.

Check:

- GPU model.
- VRAM.
- Available GPU hours.
- Session duration.
- Storage persistence.
- Dataset access.
- Model download requirements.
- CUDA/PyTorch compatibility.
- Commercial-use restrictions.
- Idle expiration.
- Billing and overage behavior.
- Hard spending limits.

No unauthorized paid compute.

Do not activate subscriptions, paid resources, automatic top-ups, or billable overages.

Do not treat a spending alert as a hard cap.

If credits or platform access require founder action, report the exact action without requesting passwords or verification codes.

Respect platform quotas and terms.

Do not evade limits through multiple accounts or prohibited background processing.

If no legitimate free GPU is available, continue all model-independent engineering and preserve the training experiment as blocked.

---

# 10. BUILD PORTABLE TRAINING INFRASTRUCTURE

Create one reproducible training pipeline with portable platform adapters for:

- Kaggle.
- Colab.
- Modal.
- Standard remote Linux CUDA.

Avoid maintaining four incompatible training implementations.

The pipeline must include:

- GPU detection.
- GPU capability checks.
- VRAM detection.
- CUDA compatibility.
- PyTorch compatibility.
- Transformers compatibility.
- PEFT compatibility.
- Quantization compatibility.
- Dataset loading.
- Dataset provenance checks.
- Model revision pinning.
- Training configuration.
- Gradient accumulation.
- Gradient checkpointing where useful.
- Optimizer selection.
- Adapter checkpointing.
- Resume after interruption.
- Controlled evaluation.
- Metric logging.
- Checkpoint hashes.
- Model export.
- Temporary resource cleanup.

Validate actual gradient updates rather than merely loading a model.

For Kaggle T4, do not assume native BF16 support.

Use a genuinely supported FP16 or alternative training strategy.

Do not assume that two T4 GPUs behave like one 32GB GPU.

Measure real memory and step time before committing to a larger experiment.

---

# 11. FINE-TUNING STRATEGY

Prepare real LoRA/QLoRA capability.

Do not fine-tune blindly.

Start by measuring unchanged-base performance.

If clear weaknesses exist, identify the smallest training intervention likely to address them.

Potential experiments:

### F1 — HalaMind LoRA

Train on planning, tool selection, ambiguity, structured proposals, and recovery.

### F2 — HalaType LoRA

Train on GUI grounding, target selection, drag-and-drop, typed actions, and Arabic interfaces.

### F3 — Multitask LoRA

Train a single adapter combining Mind and Type examples.

### F4 — HalaReliance specialization

Train a very small routing/decision classifier on verified task-routing labels.

### F5 — Adapter comparison

Compare base-only, shared LoRA, separate LoRAs, and tiny decision routing.

Use held-out evaluation.

Measure whether the added complexity genuinely improves final task success and cost.

### Important quantization distinction

Quantization and QLoRA are not interchangeable.

- LoRA modifies a small adapter during training.
- QLoRA uses a quantized frozen base during adapter training.
- Post-training quantization compresses a model for inference.
- Quantization alone does not create new intelligence.

Verify model-specific guidance before choosing 4-bit training, especially potential Qwen3.5 quantization quality concerns.

Do not assume an adapter trained under one precision/runtime is compatible with every quantized inference runtime.

Record adapter compatibility and quality explicitly.

---

# 12. MAKE HALA EXTREMELY COMPACT

The goal is to minimize:

- VRAM.
- Resident weights.
- Disk footprint.
- GPU seconds.
- Network overhead.
- Inference calls.
- Startup time.
- Hosting cost.

Investigate:

- 8-bit inference.
- 4-bit inference.
- Supported AWQ/GPTQ alternatives.
- GGUF only when multimodal processor and serving support are verified.
- KV-cache compression.
- Prefix caching.
- Efficient screenshot encoding.
- Selective image crops.
- Structured observations.
- Short output tokens.
- Reduced prompt context.
- Batching.
- Efficient model loading.
- CPU-capable HalaReliance.
- Conditional escalation.

Produce quantified results for each candidate.

Test compression effects on:

- Arabic understanding.
- GUI targeting.
- Tool selection.
- Abstention.
- Structured JSON validity.
- Recovery.
- End-to-end success.

Never claim 4-bit accuracy is equivalent to BF16 without paired evidence.

Do not force a specific file-size target without checking total model, tokenizer, processor, vision and runtime requirements.

A small weight file is not automatically an inexpensive serving system.

---

# 13. DATASET FACTORY

Implement actual dataset infrastructure.

Suggested modules:

- Dataset schemas.
- Synthetic application fixtures.
- Synthetic screenshot generation.
- Tool trajectory representation.
- Rights and provenance validator.
- Deduplication.
- Train/dev/test splitter.
- Label validation.
- Coordinate normalization.
- Task difficulty classifier.
- Arabic and RTL validation.
- Dataset version manifest.

Every applicable example should include:

- Task goal.
- Observation.
- Optional image.
- Accessibility/DOM context.
- Allowed actions.
- Correct proposal.
- Expected outcome.
- Independent verification.
- Source.
- License.
- Application/task family.
- Split identity.

Do not leak test tasks into training.

Do not confuse duplicated variants with independent evaluation samples.

Avoid personal, medical, credential, or sensitive user information.

Store approved large artifacts remotely, not in Git.

---

# 14. EVALUATION FRAMEWORK

Build a model-independent evaluation harness.

Separate:

- Model input adapter.
- Observation builder.
- Typed proposal validator.
- Synthetic executor.
- Independent postcondition checker.
- Metrics collector.
- Results serializer.

### E0 — Deterministic baseline

Validate fixtures, allowed actions, and independent success checkers without model inference.

### E1 — English/Arabic intent

Start with at least 60 frozen bilingual tasks.

Measure understanding, correct tool selection, schema validity, abstention, and ambiguity handling.

### E2 — GUI grounding

Prepare at least 120 independently generated tasks covering:

- Browser screens.
- Windows control fixtures.
- Arabic UI.
- English UI.
- RTL.
- DPI changes.
- Similar-looking controls.
- Stale observations.

### E3 — Multi-step tasks

Evaluate realistic, bounded synthetic workflows.

Record complete success, failed steps, intervention, incorrect actions, retries, and final-state mismatch.

### E4 — Tiny decision routing

Compare deterministic rules, HalaReliance, and routing through the larger model.

Measure:

- Correct route rate.
- False routing.
- Extra latency.
- CPU/GPU cost.
- Missed escalation.
- Unnecessary escalation.
- Cost per verified successful task.

### E5 — Fine-tuning effectiveness

Compare frozen-base and genuinely trained adapters under identical independent test conditions.

### E6 — Quantization

Compare model precision/runtime variants on the same evaluation cases.

All benchmark results must include exact revisions, task counts, evaluation conditions and uncertainties.

Do not use vendor scores as if they were measured Hala scores.

---

# 15. ALADDIN-COMPATIBLE ACTION PROTOCOL

Hala must communicate using bounded, typed proposals.

Study Aladdin's current proposed Hala request/response schemas.

Preserve:

- Request IDs.
- Task IDs.
- Model revision.
- Prompt revision.
- Serving configuration revision.
- Device and session binding.
- Observation generation.
- Allowed action vocabulary.
- Method ceiling.
- Explicit confidence/abstention.
- Timeout/error semantics.

Suggested proposal categories:

- `plan`
- `tool_select`
- `semantic_invoke`
- `set_value`
- `click`
- `type`
- `key`
- `scroll`
- `wait`
- `observe`
- `ask_user`
- `abstain`
- `done_proposed`

These are candidate Hala-side logical labels, not authorization to add new Aladdin kernel operations.

Map only to already qualified Aladdin capabilities.

The model must never produce a trusted completion receipt.

A model's `done` prediction means it recommends checking final state, not that the task succeeded.

Build strict validators and negative tests.

Reject unknown fields, stale references, unsupported tools, and privilege escalation attempts.

Use independent local verification for outcomes.

---

# 16. BUILD THE HALA INFERENCE SERVICE

Create one unified service exposing the three component responsibilities.

The externally visible product remains Hala.

Do not require three production endpoints or three dedicated GPU servers.

Initially support:

- `/health`
- `/ready`
- `/models`
- `/v1/hala/plan`
- `/v1/hala/action`
- `/v1/hala/decision`

Endpoint names may be adjusted to match the reviewed contract.

These are Hala service interfaces, not direct privileged Aladdin operations.

Include:

- API version.
- Authentication.
- Input schema validation.
- Size limits.
- Request timeout.
- Cancellation.
- Concurrency limits.
- Model metadata.
- Bounded output.
- Typed errors.
- No-secret logging.
- Usage and inference timings.
- Structured abstention.
- Safe degraded behavior.

Implement a mock backend for CI without GPU.

Implement a separate real-model backend requiring explicit runtime qualification.

Use a lightweight server.

Study vLLM, SGLang, and Transformers without assuming all models support identical vision or tool APIs.

Choose the smallest practical serving architecture.

Provide a reproducible container and operational documentation.

Do not deploy production inference infrastructure without a separate approved decision.

---

# 17. REAL COST MODEL

Build an executable cost calculator.

Compare:

- One 4B model.
- One 2B model.
- One shared Mind/Type model with adapters.
- Shared model plus tiny HalaReliance.
- Separate specialist models.
- Selective escalation to a larger model.

Model:

- GPU type.
- VRAM.
- Warm/cold startup.
- Requests per task.
- Images per task.
- Tokens.
- Execution retries.
- Latency.
- Idle worker expense.
- Storage.
- Network transfer.
- Concurrency.
- Queueing.
- Success rate.
- Failure cost.
- Cost per verified task.

Evaluate potential Pro user counts of:

- 10.
- 50.
- 100.
- 500.
- 1,000.

Analyze prices including $5, $8, $10, $15 and $20/month as research scenarios.

Do not adopt product pricing without founder approval.

The immediate goal is to make the model as cheap as possible while preserving practical quality.

Do not add HalaReliance to every step if doing so increases net operating cost.

Benchmark whether routing reduces total expensive-model work.

---

# 18. SECURITY AND PRIVACY

All AI model outputs are untrusted proposals.

Aladdin remains the only authority for actual computer operations.

Hala must not bypass:

- User approval.
- Device identity.
- Workspace restrictions.
- Local policy.
- Execution ceilings.
- Protected surfaces.
- Task cancellation.
- Fresh observation checks.
- Action-specific security requirements.

Build adversarial tests for:

- Prompt injection.
- Malicious webpage text.
- Screenshot-embedded instructions.
- Forged tool outputs.
- False completion claims.
- Unauthorized requests.
- Data exfiltration proposals.
- Malicious file names.
- Stale GUI references.
- Unbounded loops.
- Model output schema manipulation.
- Cross-tenant state leakage.

Default to no training on customer data.

Do not introduce third-party telemetry.

Keep all synthetic task environments isolated.

Model security must supplement, not replace, deterministic Aladdin security.

---

# 19. MANDATORY JEV, PSTACK AND GRAFT

The founder explicitly requires:

**Jev**

**PStack**

**Graft**

These are mandatory engineering and review tools.

Do not merely mention them in documentation.

Verify their actual availability, official interfaces, versions and supported capabilities.

Do not invent commands.

Use each tool genuinely for the validation and review functions it actually provides.

### Jev

Apply genuine Jev semantic review to applicable source changes.

Address correctness, security, compatibility, reliability and evidence concerns.

### PStack

Use genuine PStack review/qualification procedures supported by the installed version.

Capture findings and corrective actions.

### Graft

Run supported Graft isolated validation, testing, or change-qualification workflows.

Record genuine results.

### Per-PR evidence

Every meaningful engineering PR must contain:

- Exact code scope.
- Tests executed.
- Jev execution evidence.
- PStack execution evidence.
- Graft execution evidence.
- Findings.
- Corrections.
- Retest results.
- Blocking limitations.

Never invent passing tool results.

If a required tool is unavailable, investigate installation and repair using permitted resources.

If it remains unavailable, record the exact blocker and leave mandatory qualification incomplete.

Do not falsely substitute a handwritten report for a real tool execution.

Use ordinary static analysis, unit tests, integration tests and E2E tests as additional controls.

Use Alibaba Open Code Review where available and suitable, but distinguish delegation from a genuine managed model review.

---

# 20. REPOSITORY STRUCTURE

Initialize a lean, professional ML engineering repository.

Suggested organization:

    Hala/
      README.md
      AGENTS.md
      LICENSE
      pyproject.toml
      .gitignore

      docs/
        MASTER_PLAN.md
        CURRENT_STATUS.md
        ROADMAP.md
        DECISIONS.md
        ARCHITECTURE.md
        ALADDIN_INTEGRATION.md
        COST_MODEL.md
        FREE_GPU_RESOURCES.md
        REVIEW_EVIDENCE.md

      research/
        MODEL_COMPARISON.md
        HOLO_TRAINING_ANALYSIS.md
        LAVAGUE_ASSESSMENT.md
        DATASET_PROVENANCE.md
        QUANTIZATION_RESEARCH.md

      hala/
        mind/
        type/
        reliance/
        contracts/
        models/
        inference/
        serving/

      training/
        configs/
        scripts/
        platforms/

      data/
        schemas/
        generators/
        validators/

      evaluation/
        tasks/
        fixtures/
        metrics/

      tests/
      experiments/
      model_cards/

Create only meaningful files and directories.

Do not create placeholder-heavy documentation.

Use pinned dependencies and explicit runtime versions.

Make all code reproducible.

Keep credentials out of Git.

Document setup, testing, evaluation, training, checkpoint handling and serving.

Preserve applicable third-party license notices.

---

# 21. AUTONOMOUS IMPLEMENTATION ROADMAP

Continue through these milestones.

## H0 — Repository bootstrap

Initialize Hala, engineering guidelines, tests, CI and review workflow.

Verify Jev, PStack and Graft availability.

## H1 — Model foundation

Inspect official weights, licenses, architecture, processor and runtime requirements.

Select provisional checkpoints and immutable revisions.

## H2 — Component contracts

Implement HalaMind, HalaType and HalaReliance interfaces.

Build schema validation and mock inference.

## H3 — Evaluation harness

Build genuine synthetic task execution and independent final-state verification.

Implement bilingual tasks and GUI fixtures.

## H4 — Free GPU qualification

Check Kaggle, Colab, Modal and legitimate free alternatives.

Measure actual GPU availability and runtime suitability.

## H5 — Unchanged-model baseline

Run real inference and evaluate the first small models.

Record memory, latency, task quality and errors.

## H6 — Holo and LaVague research integration

Complete source-grounded training/data research and incorporate useful methods.

Do not import unsafe generated-code execution.

## H7 — Data factory

Generate, validate, split and version Hala training examples.

## H8 — HalaMind and HalaType adaptation

Train LoRA only when evaluation identifies meaningful correctable deficits.

Compare unified and specialized adapters.

## H9 — HalaReliance

Evaluate deterministic routing and tiny-model alternatives.

Train a tiny classifier if it materially helps.

## H10 — Quantization

Benchmark 8-bit and 4-bit inference variants with correct multimodal runtime support.

## H11 — Optimized inference

Implement bounded serving, caching, batching, resource controls, and model readiness.

## H12 — Aladdin integration readiness

Publish versioned API contracts, mock service, example payloads, adapters and integration tests.

## H13 — Release qualification

Produce a reproducible Hala inference package, model cards, measured evaluations and an explicit release-readiness report.

No milestone may be marked complete without real qualifying evidence.

When paid compute or other external authorization blocks a milestone, complete all independent work and preserve the blocked status.

---

# 22. CONTINUATION RULES

This is not a one-PR research task.

Continue progressing the actual project.

After each grain:

1. Implement the work.
2. Add tests.
3. Run relevant validation.
4. Run Jev.
5. Run PStack.
6. Run Graft.
7. Fix real findings.
8. Re-run affected tests and reviews.
9. Commit the changes.
10. Open or update an appropriate PR.
11. Merge only when authorized and qualified.
12. Verify canonical main.
13. Update project status.
14. Continue with the next legitimate engineering task.

Do not stop merely because initial scaffolding passed.

Do not stop after writing documentation.

Do not claim that mock inference is a trained model.

Do not claim a successful model benchmark without actually loading and testing the checkpoint.

Do not claim zero inference cost based on synthetic calculations.

Do not repeatedly ask the founder to approve routine nonprivileged engineering tasks already within scope.

Never interpret autonomous continuation as permission to spend money, access private resources, bypass review, or change Aladdin's authority.

---

# 23. DEFINITION OF COMPLETION

Hala is genuinely integration-ready only when:

- The repository is professionally structured and tested.
- HalaMind is implemented and evaluated.
- HalaType is implemented and evaluated.
- HalaReliance is implemented or has a documented evidence-based no-model decision.
- A real open-weight multimodal checkpoint has been qualified.
- Arabic and English tests have been executed.
- GUI grounding has been evaluated.
- Tool proposals have been validated.
- Independent task-completion tests have been run.
- Quantization has been tested where supported.
- Any fine-tuned adapter has real training provenance.
- GPU memory and latency have been measured.
- Cost models incorporate genuine compute data.
- Inference-server contracts are functional.
- Aladdin-compatible mock integration passes.
- Security tests pass.
- Required Jev, PStack and Graft reviews have authentic evidence.
- Model/license requirements are satisfied.
- Reproducible artifacts and operational instructions exist.

A real production launch has additional infrastructure, signing, commercial-rights, security and founder-approval gates.

Do not mark production ready before those gates pass.

---

# 24. FINAL REPORTING

For every major milestone, report:

- Canonical GitHub main SHA.
- Current branches and PRs.
- Merged changes.
- Implemented capabilities.
- HalaMind status.
- HalaType status.
- HalaReliance status.
- Chosen base checkpoint.
- Exact model revision.
- Actual training runs.
- Adapter artifacts.
- Quantization outcomes.
- GPU used.
- VRAM.
- Measured latency.
- Verified task-success rate.
- Arabic evaluation.
- English evaluation.
- Jev findings.
- PStack findings.
- Graft findings.
- CI status.
- Out-of-pocket spending.
- Remaining technical blockers.
- Next executable task.

Keep `docs/CURRENT_STATUS.md` and `docs/ROADMAP.md` accurate.

Clearly distinguish:

- NOT STARTED
- IMPLEMENTED
- TESTED
- QUALIFIED
- BLOCKED
- PROPOSED

Never report training success, deployed inference, or release completion without evidence.

---

# FINAL FOUNDER INSTRUCTION

I am building **Hala** to become the native intelligence of **Aladdin Pro**.

Claude is already working on Aladdin.

Your responsibility is to develop Hala independently and in parallel.

The three official components are:

**HalaMind** — reasoning, planning and task intelligence.

**HalaType** — GUI understanding, visual grounding and typed computer actions.

**HalaReliance** — very small, fast and cost-efficient task decisions.

I want Hala to be:

- Small.
- Fast.
- Powerful.
- Multimodal.
- Excellent in Arabic and English.
- Capable of Windows and browser tasks.
- Reliable at selecting tools.
- Efficient with GPU resources.
- Commercially usable.
- Easy to integrate with Aladdin.
- Inexpensive to operate.

Study Hcompany Holo training and its publicly released suitable datasets.

Study LaVague's World Model, Action Engine, context retrieval, QA and deterministic navigation.

Use the best available compact open-weight foundations.

Develop real LoRA/QLoRA training infrastructure.

Investigate genuinely useful specialized adapters.

Quantize and benchmark models without assuming quality is preserved.

Use Kaggle, Colab, Modal and other legitimate free GPU opportunities before paid resources.

**Target $0 out-of-pocket expense for initial research, training and engineering.**

No unauthorized paid GPUs, subscriptions, top-ups or billable infrastructure.

Do not download multi-GB model weights to my personal computer.

Store approved datasets and model artifacts remotely.

Use **Jev, PStack and Graft genuinely and consistently**.

Keep all source code and documentation in:

https://github.com/TheHalfMoon/Hala

Do not modify Claude's Aladdin work.

Do not stop after a single milestone.

Continue autonomously through every safely authorized research, implementation, training, evaluation, optimization and qualification task.

Maintain truthful evidence.

**The goal is a real, efficient, tested, compact Hala AI system—not a collection of plans or merely a rebranded base model.**

**ENGLISH ONLY.**

**VERIFY THE LIVE REPOSITORY AND START BUILDING NOW.**
