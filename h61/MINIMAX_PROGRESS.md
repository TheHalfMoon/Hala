# H61 / MiniMax — complete the original corpus, capability-balanced curriculum, real grounding training

**Branch `ps/h61-complete-corpus-and-curriculum`, stacked on the H60 handoff
`3e97a9f` (Draft PR #44), which is stacked on the H59 handoff `1746592`
(Draft PR #43).** All counts below are measured, not advertised. "DONE" always
carries a receipt.

## What was verified by this agent before changing anything

| Fact | How it was verified |
|---|---|
| The live GitHub frontier is **PR #44 (H60)**, not PR #43 | `gh pr list` returned PR #44 open at 2026-10-21T16:19:33Z with 13 commits, stacked on `ps/h59-minimax-million-data-plan` |
| H60 CI | PR #43 shows 4/4 `SUCCESS`; PR #44 shows 4 queued at the time of reading |
| The H60 worktree is **actively used by a concurrent agent** | commit `3e97a9f` appeared during this session and `h60/ingest.py`, `h60/publish.py`, `tests/test_h60_publish.py` changed at 19:42-19:43 local, i.e. concurrently |
| The three founder private Kaggle datasets exist and are private | authenticated `datasets/view` on `hala-million-source-map-private`, `hala-gui-training-library-private`, `hala-dragon-multidomain-train-private` returned `isPrivate=true` |
| The four H60 slider shard datasets exist and are private | `datasets/view` on `hala-h60-dragon-train-slider-0000..0003-private` returned `isPrivate=true`, `totalBytes` ≈ 1.076 GB each |
| The H60 shard **byte-level readback proof is not durable** | `h60/staging/shard-receipts/` is empty and the provider copy of a 28,806 member shard is not downloadable; the concurrent agent replaced the packaging with one opaque member for exactly this reason |
| Kaggle GPU eligibility is blocked | the H60 ledger records the inspected Terms of Use §3 non-commercial restriction; **the guards were kept, not weakened**, and no Kaggle GPU session was launched |
| Vercel Workflow is absent | no `vercel` CLI, no credential, no deployment; nothing is claimed |
| The free GPU allocation is a **single** Colab T4 | `colab new --gpu T4` was refused with `precondition failed` while the concurrent agent's `hala-h60` session held the allocation, and `colab usage` reports `balance 0.00 compute units` before and after |

## Corrections to the inherited ledger

| Inherited claim | Measured state | Correction |
|---|---|---|
| "the complete original `train/slider.tar` is byte-exact" | the local range receipts cover **6 windows** and the local shards hold **57,797 of 100,000** original screenshots and **506,857 of 877,709** native actions | the slider archive was **partially** materialised; the whole-archive SHA-256 proof did not complete durably. H61 does not repeat the claim. |
| "the other three original TRAIN archives: IN PROGRESS" | nothing of `sheet`, `slide_resize` or `text_highlight` was ever downloaded | still true; H61 ingests them |
| "the advertised 65,000 / 571,350 slider count understates the archive" | the four local shards hold 57,797 images / 506,857 actions, which is **consistent with** the advertised count being about 60 % of the archive | the correction is not disproved, but it is not yet proven either; it needs the whole archive |

## H61 diagnosis of the measured H60 negative result

The H60 adapter learned the output contract (strict bare-JSON validity
0 → 17 of 48) and then collapsed onto a constant degenerate point
(geometrically correct 0 of 24 on both public DEV sets, real GUI drag task
success 0 of 48). Three concrete, testable causes were identified:

1. **Supervision dilution.** The completion is a JSON object whose tokens are
   dominated by the deterministic structural keys; the coordinate digits are a
   small fraction. A token-mean cross entropy therefore spends most of its
   gradient on tokens that are already predictable, and the cheapest way to
   reduce the mean loss is to predict a plausible constant coordinate. H61
   raises the weight of the coordinate tokens (`h61/supervision.py`).
2. **Train/evaluate pixel-frame mismatch.** The H60 curriculum trained on the
   original 1280x800 DragOn screenshots, while the public DEV evaluation
   frames are 1024x640, and the upstream official evaluation screenshots are
   1280x800. A model that learned absolute coordinates in one frame has to
   generalise across frames to be scored in another. H61 moves the training
   screenshots into the evaluation frame and scales the original boxes with
   the original screenshot (`h61/source_export.py`, `h61/official_eval.py`).
3. **Prompt-contract drift.** The H60 curriculum prompt and the H60 public
   DEV prompt were two similar-looking templates with nothing proving they
   were the same contract. H61 has one template (`h61/prompt_contract.py`)
   and a test that asserts the equivalence against the H60 runner itself.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| Verify the live frontier, the concurrent work and the provider state | **DONE** | the table above |
| Bounded, verified export of original TRAIN material | **DONE** | `h61/source_export.py`; one measured export produced 542 original screenshots and 4,789 native drag actions with 0 rejections, every fetched range carrying a SHA-256 receipt |
| The task-region offset resolved with a bounded probe | **DONE** | the measured cost of the inherited wide probe was minutes of source bandwidth per domain; the bounded 256 KiB probe resolves the same offset `6231426560` for the slider archive in 28.6 s |
| Capability-balanced curriculum with screenshot deduplication | **DONE** | `h61/curriculum.py`; balanced per capability, capped per screenshot, deterministic, create-only, reload refuses a modified export |
| One shared prompt contract for training and evaluation | **DONE** | `h61/prompt_contract.py` and `tests/test_h61_prompt_contract.py` |
| Coordinate-focused supervision | **DONE** | `h61/supervision.py`; the weighted loss provably reduces to the plain token mean when every weight is one |
| Independent benchmark source qualified | **DONE** | the upstream `eval/` split was found, fetched and its task format confirmed identical to `train/`; 12 official scenes were fetched and scored locally |
| Real training run on the free T4 | **QUEUED** | the single free GPU allocation is held by the concurrent H60 run; the H61 run waits for it and starts itself, it never competes for the allocation and never starts a duplicate session |
| Real matched public DEV measurement of the H61 adapter | **QUEUED** | same runner, same cases, same prompts, same scorers as the frozen base |
| Real matched official-split measurement of the frozen base and the adapter | **QUEUED** | `h61/official_eval_run.py` |
| Private Kaggle mirror of the complete corpus | **OWNED BY THE CONCURRENT H60 WORK** | H61 does not upload a second copy of the same corpus; it consumes the original source directly with byte-range verification, so no dataset is published twice |
| Checkpoint persisted to authorised private storage | **PENDING** | the training run publishes an immutable checkpoint bundle at every step; the bundle is downloaded and pushed to a private provider dataset after the run |
| Kaggle GPU | **BLOCKED** | the inspected provider terms restrict service use to internal, personal, non-commercial use. The guards stay. |
| Vercel Workflow | **NOT DONE** | no account, credential or deployment exists in this environment. The durable orchestration is portable inside `h61/colab_run.py` and `h59/durable_contract.py`. |

## Honest limits

* No H61 model result exists yet. Every training and evaluation number in this
  ledger is a target, not a measurement, until the run finishes and the raw
  receipts are read back.
* The upstream official evaluation split is public and is used for evaluation
  only. It is never read into a training row and never used to derive a label.
* The official evaluation is measured in the H61 evaluation frame, which is a
  documented, exactly reversible uniform scale of the official 1280x800
  screenshots and of the official start and end boxes.
* A drag on an unordered original task is accepted in either direction
  **because the original task says the direction does not matter**. An ordered
  original task is never accepted in the reversed direction.
* Nothing here claims a competitive or state-of-the-art result.
