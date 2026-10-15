# H18: shared Mind/Type adaptation preparation

Status: preparation TESTED; independent parent correctness/integrity reviews and repair deltas passed. Genuine Jev/Graft gates passed; integrated153 CPU tests passed. Prospective quality training remains disabled.
Scope: deterministic data/contract readiness audit and disabled experiment specification.
This phase neither trains nor qualifies a learned model.

## Resolved choices

Retain one pinned Qwen3-VL-2B multimodal base. First compare its unchanged behavior
under task-specific Mind and Type prompts. If suitable new data and evidence justify
adaptation, propose one rank-4 multitask LoRA on that shared base. Separate adapters
require a later measured deficit, suitable data, and a demonstrated gain large enough
to justify residency, serving and maintenance costs. No split-adapter work is authorized
here. The current 58-row artifact is inventory evidence, not the proposed training set.

## Acceptance and evidence

- [x] Audit original frozen source allocations, candidate rows, all media hashes/rights,
  family/language counts, source identities and exact original prompts/first actions.
- [x] Distinguish pixel/window logical actions from normalized point outputs and bound
  Mind plans. Report English-only coverage and unqualified Arabic rendering.
- [x] Recompute H14 per-family/language outcomes from frozen tasks and raw outputs;
  enforce its training exclusion and report H15's limited engineering evidence.
- [x] Reject altered rows even with updated JSONL hashes, relabeled heldouts/languages,
  missing rights, altered media, rewritten H14 scores and unsafe config overrides.
- [x] Pin reviewed historical examples/media catalog, original source, H14 task/freeze/raw
  result and H15 report/independent record. Reject coherent image+digest substitutions,
  fabricated raw outputs with consistent scores, and false scope/Arabic/quality claims.
- [x] Produce reproducible JSON evidence and one disabled shared-model configuration;
  absent compatible data, independent splits and reviewed runtime keep training blocked.
- [x] Full CPU suite once; exact result recorded in docs/H18_ADAPTATION.md.
- [x] Independent PStack, genuine Jev and supported Graft source qualification by parent; no release or learned-quality approval.

## Hard stops

No model downloads, training, jobs, paid compute, native execution, customer data,
or automatic conversion of the existing artifact. H14 remains permanently held out
from training and prompt tuning. Do not use H15 loss/update/reload measurements as
evidence of learned quality. No H14/H15/H17 or production contract changes.

## Future data admission requirements

Every proposed dataset needs immutable bytes, image hashes, reviewed provenance and
commercial rights for labels and media; no benchmark or customer material. Use original
new tasks with independent, family-disjoint train/dev/test allocations frozen before
training. Detect duplicated IDs, images, normalized goals and underlying task templates
across splits; language translations and layout variants belong in the same allocation.
H14 tasks, images, variants, private labels and benchmark trajectories cannot enter any
training or prompt-development set. Existing synthetic dev/test fixtures are useful
engineering tests; they are not compatible adaptation dev/test labels.

Mind data must supervise `hala.plan.v1` against the actual bound `Request`, using
`Plan.from_dict` and deterministic fixture state execution for acceptance. Include
bounded decomposition, tool choice, ambiguity/ask-user, cancellation, abstention and
recovery. Only observed controls and trusted request bindings may be referenced;
never forge authority, receipts, or verified completion. Current first-action labels
cannot be converted into plans by wrapping them or inventing missing steps.

Type data must pair a goal and entire-frame screenshot with exactly `{"point":[x,y]}`
integer 0..1000 labels, using the fixed H14/H16 mapping semantics and half-open target
geometry. Its model input must exclude private targets, answer rectangles and incidental
fixture IDs. Preserve downstream H16 action/request validation and local execution
authority. A new reviewed converter and actual tokenizer/processor masking audit are
required; pixel/window/ref-hint completions are not normalized labels.

Both tasks need English and Arabic semantic cases and hard distractors, including
numeric comparison and ordering in independently authored scenarios. Verify Arabic
glyph shaping/RTL rendering with actual RAQM and declared font; visual inspection and
image freezes precede model loading. A manifest flag alone is insufficient. Determine
the smallest defensible sample size and family count from dev evidence and uncertainty;
this phase does not invent training counts or generate a large corpus.

## Prospective experiment, blocked until reviewed

1. Freeze new compatible dataset, split/rights/duplicate reports, task prompts and
   evaluation code. Measure unchanged pinned base separately on Mind and Type in both
   languages. H14 motivates Type investigation (especially semantic comparisons); it
   supplies no Mind baseline and does not justify claiming an Arabic-only deficit.
2. Pre-register a bounded free-runtime budget, exact dependencies/modules, task mixing,
   rank/optimizer/step cap, processor/image/token bounds, checkpoint retention, selection
   rule and stopping criteria. Candidate hyperparameters in JSON are inherited small
   H15 starting values, not a training recipe or a qualified optimum. Use dev only for
   bounded selection; no test prompt/model tuning. Gate before model allocation.
3. If unchanged-base deficits and data support it, compare a single shared multitask
   LoRA against the same unchanged base with fixed prompts and sampling. Evaluate Mind
   with independent fixture state and safety/abstention checks, Type with fixed geometry
   and independent state transitions. Report schema validity, wrong targets, misses,
   complete-task success by task/language/family, uncertainty and family dependence.
   Measure actual warm/cold latency, memory, calls and cost per verified success.
4. Select no adapter if gain is absent, uncertain or causes unacceptable safety/language
   regression. Freeze selection thresholds and uncertainty treatment before test use.
   Run independent adaptation test once after dev selection. A final read-only H14
   regression comparison may be separately authorized after candidate freeze; it cannot
   select or tune the candidate. Repeated H14 inspection is not a fresh independent test.
5. Require independently reviewed quality, runtime, retention and release evidence for
   any later adoption. No serving/deployment or founder merge permission is implied.

The H18 implementation intentionally lacks a generic compatible-data loader and a
training runner. Supplying files or toggling Boolean claims cannot make it ready;
later admission requires a separately reviewed implementation and experiment.
