# Training repair delta review 1

Immutable artifact: `work/training-final-review.diff`.
SHA-256: `d5d9d225a044ca9c0b278b76738fc168fb8358f8b45d9b068fe0eeeb9deeeb12`.
Reference: `docs/reviews/foundation-combined.md` (C1–C4 and C5/P1/C6). No source edits during review. All 21 added text files match this snapshot; generated images were checked through the matching manifest's media digests.

**Verdict: one must-fix remains (C3); C1/C2/C4 are resolved at source/helper level.**

| Finding | Result | Concrete evidence |
|---|---|---|
| C1 media integrity | Resolved | `training/artifacts.py:110–187` binds exact JSONL bytes, every media path/digest/family/provenance, bounded rows, PNG headers and token budget. `train_adapter.py:76` performs the complete audit before the runtime probe at line 93. All 58 English media records load. Independently replacing the **last** media item fails with a digest error before the runtime probe; completed resume also rejects changed media. |
| C2 restart integrity | Resolved | `training/checkpoints.py:117–150` writes adapter/optimizer state into a new version, verifies the exact hashed file set, syncs it and publishes one atomic pointer. The old version is untouched. Existing tests cover staged write/rename interruption, corruption and mismatched identities. Independent failure injected at the actual `os.replace` of `latest.json` preserves the prior complete pair. `train_adapter.py:86–92` finalizes a completed checkpoint before the probe/model imports; an independent synthetic checkpoint retained both saved metrics and update evidence, with no probe call or new torch/peft/transformers module. Partial resume retains metrics, restores optimizer/scaler/RNG and verifies identities in source. |
| C3 finite economics | **Must-fix remains** | `experiments/cost/calculator.py:65–66,75`: magnitude cap `1e12` rejects extreme large inputs, but tiny positive denominators remain valid. `success_rate=5e-324` produces `usd_per_verified_task=inf`; `month_hours=5e-324` produces `offered_concurrency_load=inf`. Both then fail `json.dumps(..., allow_nan=False)`. **Fix:** reject any nonfinite derived result before returning; add these two regression inputs. |
| C4 diagnostic JSON | Resolved | `training/scripts/baseline.py:20–38,135–141` parses raw answers strictly before point validation. Independent probes reject duplicate keys (including nested), nonfinite constants and exponent overflow; valid integer point output passes. |
| C5/P1/C6 | Addressed within candidate scope | PNG dimensions are checked before RGB conversion; exact bounded row fields and a declared processor-expanded token budget exist. Candidate terminology replaces unearned measured/qualified claims. Completion-boundary helper masks the entire prefix and rejects mismatched/full-length/over-budget prefixes in CPU tests/probes. Actual processor behavior remains unexecuted. |

Verification: **83/83 full CPU tests passed** using bundled Python through `rtk`; narrow independent cost/baseline review ran **7/7 targeted tests** plus boundary probes. No account-limit blocker prevented this review. The independent probes used synthetic checkpoint bytes, not real adapter or optimizer tensors.

Correctness remains blocked only by the demonstrated C3 numerical defect. The added artifact/mask/checkpoint modules have concrete boundary roles and do not create a parsimony must-fix. Product/source claims consistently identify a candidate pipeline: the converter prepares 58 original English initial-state PNG/prompt/logical-action records, excludes Arabic images until separate shaping qualification, holds out dev/test families, and marks training unexecuted and adapter unqualified. The tested `LORA_TARGET_PATTERN` constant matches sample language projections and excludes sample vision projections; this review makes no regex-bug claim or actual-module qualification claim.

Limits: no local foundation weights, remote job, actual processor invocation, LoRA step, model-fit evidence, continuous/resumed numerical comparison, adapter quality or model qualification. CPU fault recovery verifies publication logic only. The prior separate security gate is not replaced by this combined delta review.

panel: delta — correctness ✓ independent source/tests/mutations (C3 remains) · parsimony ✓ inline bar · product ✓ source status · security — prior separately coordinated review; no actual processor/LoRA/model qualification
