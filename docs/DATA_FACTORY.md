# Original synthetic data and evaluation

Implemented and CPU-tested on 2026-10-14 with Python 3.12.14. This qualifies the synthetic evaluation infrastructure only. No foundation model, real application executor, browser automation, image grounding, training run or production deployment has been qualified by these results.

## Frozen source fixtures

`data/frozen/tasks.json` contains 60 intent fixtures (30 English, 30 Arabic) and 120 GUI fixtures (60 English, 60 Arabic). Seed `20261014`, generator revision `hala-synthetic-v1`, and canonical UTF-8 JSON serialization reproduce the file byte-for-byte. The SHA-256 file detects drift; it is an integrity checksum, not a signature or source authenticity guarantee.

The 180 task IDs have independently sampled document identifiers, values, control positions and DPI scales. They represent **12 workflow families**, not 180 independent workflow families. Intent and GUI tasks sharing a workflow have the same family ID and split. Pixel/layout/language variants are never counted as additional independent families.

| Split | Tasks | English | Arabic | Families |
|---|---:|---:|---:|---:|
| train | 116 | 58 | 58 | 8 |
| dev | 32 | 16 | 16 | 2 |
| test | 32 | 16 | 16 | 2 |

The suites cover save, open, archive, export, toggle, form submit, filter apply, drag/drop, transient recovery, scroll/confirm, rename and deletion confirmation. GUI fixtures include similarly named distractor controls; Arabic layouts mirror control geometry. Frame scales are 1, 1.5 and 2. Each bounding box has captured-pixel and normalized coordinates. Normalization uses the captured frame dimensions; dividing by DPI again would be wrong.

All fixture content is originally generated synthetic material dedicated to CC0-1.0. No source screenshots, benchmark tasks, benchmark trajectories or personal records are incorporated. Source metadata must match the explicit original-synthetic allowlist. Unknown sources, benchmark flags, unsupported rights and noncommercial-training flags are rejected. Training import is restricted to exact original generator records: it regenerates the declared integer seed, maps original task IDs to full-record SHA-256 digests, and rejects unknown IDs or any content change. The default frozen source digest is pinned in the importer. Arbitrarily relabeled external material is not an accepted import source; a generic external-dataset rights-verification service is outside this implementation.

Validation rejects task duplicates, semantic duplicate tasks, duplicate controls, ambiguous exact labels, family leakage, invalid coordinates, wrong action/role labels, invisible step targets, undeclared state keys and final-state labels unreachable by the annotated steps. Task IDs bind suite, workflow and bounded index; initial and desired document identities must agree with the task ID. Semantic deduplication applies Unicode NFKC and collapses incidental goal whitespace while retaining exact state and action semantics. Annotated steps, including transient failed attempts and observations, must fit the admitted budget. A separate label replay checks reachability before fixtures are admitted. The independent evaluation checker still reads actual executor state when evaluating a provider.

## Execution and verification

`evaluation.engine` separates a proposal provider, fresh observation adapter, strict Hala contract validation, synthetic executor and final-state checker. Requests use `Request.from_dict` and `to_dict`; proposals use `Proposal.from_dict` against the current request. Responses bind to the task, session and observation generation. The executor revalidates against its own current observation and rejects stale references, wrong DPI, unsupported roles and multi-action batches. Every accepted action or rejected attempt consumes a bounded task step.

The executor mutates its own state through control effects and prerequisites. It does not read the desired state to apply effects. The runner detaches a task snapshot before calling the provider, so later mutations to a provider-retained caller object cannot change verification labels, budget or result identity. The final-state checker compares actual state to that original snapshot's desired state; provider postconditions and `done_proposed` claims do not establish success. Wrong text, unrelated buttons and premature completion therefore fail verification. Success can be observed even when a provider omits a completion claim, because actual final state is authoritative.

Forms require the expected value before submit; deletion confirmation appears only after opening its dialog; scroll reveals a confirm target; recovery ignores the first click and requires a fresh observation and retry. Drag/drop is explicitly a **semantic pick/drop abstraction** implemented using two `semantic_invoke` actions. Hala v1 has no pointer drag action, so this does not qualify physical pointer dragging or native drag-and-drop.

The `oracle` provider reads privileged fixture labels. Its perfect results establish that the task machinery is reachable and functioning; they measure no intelligence. The `mock` provider exercises Hala's existing deterministic contract mock. A future real-model provider can implement `propose(request)` without changing the checker, but the currently strict contracts accept only explicitly unqualified mock backend metadata. Real-backend qualification needs a separately reviewed contract extension.

## Reproduction

From the repository root, using Python 3.11+:

```text
python -m unittest tests.test_evaluation -v
python -m data.factory
python -m data.export --output data/frozen/train.jsonl
python -m evaluation --provider oracle --split all --output evaluation/results/oracle-all.json
python -m evaluation --provider oracle --split test --output evaluation/results/oracle-heldout.json
python -m evaluation --provider wrong_target --split all --output evaluation/results/wrong_target-all.json
python -m evaluation --provider stale --split all --output evaluation/results/stale-all.json
python -m evaluation --provider invalid --split all --output evaluation/results/invalid-all.json
python -m evaluation --provider false_done --split all --output evaluation/results/false_done-all.json
python -m evaluation --provider mock --split all --output evaluation/results/mock-all.json
```

The CLI defaults to the heldout `test` split; selecting `all` is an explicit infrastructure audit. The training exporter writes only exact original train-family fixtures and a content hash. Original source matching includes workflow, goal, family and split, so neither relabeling a heldout task's metadata nor inventing a new training ID admits its content. Dev/test fixtures and evaluation reports are never exported as training examples. `training_records` preserves its API: it returns deep copies of complete original task dictionaries. Alternate integer seeds remain reproducible and can import their own exact original generated records. `train.jsonl` has 116 fixtures in eight families, and shares no task IDs or families with dev/test.

## Actual infrastructure results

The test suite was first run before implementation and failed because `data.factory` was missing. After implementation, all 12 evaluation tests passed. An independent review then exposed four boundary defects: caller-owned label mutation, heldout relabel admission, budget-unreachable annotations and formatting duplicates. New regression tests reproduced eight failing subcases before repairs. All 16 evaluation tests now pass, including canonical source-drift/identity rejection, provider budget mutation, transient budgets, whitespace variants, fullwidth Unicode and Arabic canonical composition. Frozen fixture and training-export bytes remain unchanged. Saved JSON reports contain per-task traces and final states, bilingual and suite strata, schema failures, stale failures, wrong-target counts, false completion counts, and descriptive Wilson intervals.

| Provider | Tasks | Verified final state | Detection evidence |
|---|---:|---:|---|
| privileged oracle | 180 | 180 | All 12 workflow families reachable |
| wrong-target control | 180 | 0 | 180 wrong targets, 180 false completion claims |
| stale control | 180 | 0 | 180 stale proposal rejections |
| schema control | 180 | 0 | 180 invalid output rejections |
| false-done control | 180 | 0 | 180 false completion claims |
| Hala deterministic contract mock | 180 | 57 | 57/60 intent; 0/120 GUI |

The mock understands its limited fixture syntax: it passes 30 English and 27 Arabic intent cases and abstains on Arabic text entry and all multi-step GUI goal syntax. These are known scripted capabilities, not model performance or language-quality measurements.

The oracle heldout run reaches 32/32 tasks in **two** heldout families. The descriptive 95% task Wilson lower bound is 0.8928; the family lower bound is 0.3424. Across all source fixtures the corresponding lower bounds are 0.9791 and 0.7575. Task variants are dependent, and even the family interval is descriptive: these designed workflow families are not a random sample of real-world tasks. None of these intervals should be used to advertise model quality or generalization.

## Optional PNG rendering

```text
python -m evaluation --provider oracle --split test --output evaluation/results/oracle-heldout.json --screenshots evaluation/screenshots --font C:/Windows/Fonts/arial.ttf
```

Pillow 12.3.0 rendered 32 actual PNG initial-state fixtures in this session using Windows Arial. Each screenshot record includes its font and `rtl_shaping_supported`. This bundled Pillow lacks RAQM, so that field is **false**: mirrored Arabic control geometry and Unicode glyphs render, but contextual Arabic shaping and bidi text display are not visually qualified. A RAQM-enabled Pillow build must be provisioned and visually checked before these images are used as an Arabic visual benchmark. The current contract mock reads the structured observation; these PNGs were not fed to a vision model.

Pillow is optional and not part of CPU contract/runtime dependencies. No dependency was installed for this evaluation.
