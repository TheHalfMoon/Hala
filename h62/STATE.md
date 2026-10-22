# H62 — Durable task ledger

**Single source of truth for the H62 milestone.** One row per task, one
status each. `DONE` means the evidence exists. `ACTIVE` means it is
running right now. `BLOCKED` names the exact blocker. `NEXT` is the ordered
follow-up. Nothing is marked `DONE` on the strength of an intention.

Milestone: H62 — free-GPU acquisition, the H61 diagnostic launch, and the
broader computer-use programme.
Base: H61 `5838f0b757693a7622dc6a40818be44928eec3b3` (Draft PR #45), on H60
PR #44, on H59 PR #43.
Branch: `ps/h62-minimax-live-training-handoff` (Draft PR #46).
Last updated: 2026-10-22.

## DONE

| # | Task | Evidence |
|---|---|---|
| D1 | Verified the live GitHub frontier: PR #46 (H62) on PR #45 (H61) on PR #44 (H60) on PR #43 (H59); all four open, draft, `MERGEABLE`; `main` is a base stub | `gh pr list --repo TheHalfMoon/Hala` |
| D2 | **Verified that the inherited 36-hour allocation mechanism was NOT running.** No watcher process, no live session, zero assignments, zero balance | `Get-CimInstance Win32_Process`; `colab sessions`; `colab usage` — all in `h62/GPU_ACQUISITION.md` §1 |
| D3 | **Proved the Colab assignment path works for this account**: a CPU session on the same account, same CLI, same minute reached `Session READY` while the T4 variant returned HTTP 503 | `colab new -s hala-h62-cpu` → `Session READY`; `h62/GPU_ACQUISITION.md` §2 |
| D4 | Proved remote execution end to end: a file executed in the remote Linux kernel and its real output read back | `colab exec` → `HALA_PROBE {"python": "3.13.16", "platform": "Linux-...", "exec_ok": true}` |
| D5 | Proved zero paid compute: balance `0.00` before, during and after; usage rate returned to `0.00/hr`; the probe session was stopped cleanly | `colab usage` before/after; `h62/GPU_ACQUISITION.md` §2 |
| D6 | Measured the Kaggle accelerator quota (`GPU 28.91h remaining / 30h`, `TPU 20.00h`, refresh `2026-10-24T00:00:00`) and confirmed **no concurrent job** holds it | `kaggle quota`; `kaggle kernels list --mine` + `kaggle kernels status` on every Hala kernel → all `COMPLETE` |
| D7 | Kept the Kaggle guard. H62 does **not** train on Kaggle: the provider terms restrict use to internal, personal, non-commercial use and Hala is built for a third party. The measured quota is recorded as a fact and the founder's decision point, not silently consumed | `training/scripts/train_adapter.py`, `training/scripts/baseline.py` untouched; `docs/FREE_GPU_RESOURCES.md` |
| D8 | Built the durable bounded T4 watcher: single request at a time, adopts an existing session, never allocates on a non-zero balance, exponential backoff with jitter, hard request budget, hard deadline, `fsync`-ed JSONL receipt stream that resumes across a restart, honest failure classification, secret redaction | `h62/colab_t4_watcher.py` + `tests/test_h62_colab_watcher.py` (19 tests) |
| D9 | Built the pinned-source remote entry point: fetch by exact 40-char SHA (never by branch), verify HEAD before any weight download, require an explicit free-runtime confirmation, refuse a Kaggle notebook, pin package versions, propagate a nonzero child exit and write a failure receipt first | `h62/remote_h62_launch.py` + `tests/test_h62_remote_launch.py` (10 tests) |
| D10 | Built the local operator runner: verifies the session really has a GPU, runs the pinned entry point, keeps the full provider log, reads the receipts back | `h62/run_h61_diagnostic.py` |
| D11 | Preserved the verified H61 pipeline and its evidence untouched. No H61 source, receipt, curriculum or test was modified | `git diff` over `h61/`, `h60/`, `research/`, `reviews/` is empty |

## ACTIVE

| # | Task | State |
|---|---|---|
| A1 | **The free T4 watcher is running.** Detached, state directory on disk, receipt stream at `h62/watcher/watcher-receipts.jsonl`. It asks for a free T4 with a 120 s → 1800 s jittered backoff, a budget of 80 requests and a 36 h hard deadline, and it stops the moment a GPU session is allocated | Live receipt stream; `h62/GPU_ACQUISITION.md` §4 |
| A2 | **H62 capability and architecture investigation** into the ten required computer-use capabilities, with file-and-line evidence for what is implemented, mocked, measured or missing | `h62/ARCHITECTURE_REVIEW.md` |
| A3 | **Source-first data programme** for millions of verified action annotations across the capabilities, with original / annotated / generated / duplicate / independently-verified separation and train/DEV/TEST isolation | `h62/DATA_ROADMAP.md` |
| A4 | **Training science**: the H60 baseline preserved, matched evaluation conditions, controlled ablations for coordinate weighting, image-frame alignment, prompt consistency and bilingual sampling, and a constant-coordinate-collapse detector | `h62/TRAINING_SCIENCE.md`, `h62/collapse_probe.py` |
| A5 | **Qualification**: external computer-use and visual-grounding benchmark plan, real task success, coordinate accuracy, language breakdown, recovery behaviour and failure cases | `h62/QUALIFICATION.md` |

## BLOCKED

| # | Task | Exact blocker | What would unblock it |
|---|---|---|---|
| B1 | **The H61 diagnostic training run** | No free GPU session has been granted. The provider answers every free T4 assignment with HTTP 503. The assignment path itself is healthy (a CPU allocation succeeds on the same account in the same minute), so the blocker is the availability of the **T4 accelerator variant**, and the provider does not disclose whether that is capacity or per-account entitlement | A free T4 allocation (the watcher is live) **or** explicit Kaggle permission for commercial research use, which would make the measured 28.91 h of free GPU usable |
| B2 | The H61 matched public-DEV and official-split comparison, and any model-improvement claim | Depends on B1. There is no trained adapter to compare | B1 |
| B3 | A public model release | Not authorised, and not requested | Explicit founder authorisation after a measured, matched evaluation |
| B4 | Durable cloud orchestration | No Vercel account, credential or deployment exists in this environment, and none was created. The portability lives in `h61/colab_run.py` and `h59/durable_contract.py` instead | A founder-provisioned Vercel instance with an explicit $0 boundary |

## NEXT

1. Read the watcher's receipt stream. The moment it records an `ALLOCATED`
   outcome, run the pinned H61 diagnostic: `python -m h62.run_h61_diagnostic
   --pinned-sha <the reviewed commit> --local-dir h62/run`.
2. Publish the real receipts: the remote launch receipt, the H61 run receipt,
   the checkpoint bundle SHA-256, and the matched evaluation of the frozen
   base, the H60 adapter and the H61 adapter on the same prompts and the
   same images. Publish the outcome even if it is zero, and publish it even
   if it is worse than the frozen base.
3. Only if the grounding moved, scale the mixture and add the click, typing,
   scrolling and multistep families back beside the drag rows.
4. Turn the architecture findings into the smallest honest fixes, each with
   its own test, rather than a large speculative rewrite.
5. Admit the next source-first corpus batch under the leakage rules in
   `h62/DATA_ROADMAP.md`, without republishing anything the concurrent H60
   corpus work already published.

## Preserved negative evidence

The only measured real GUI result in this project stays on the record and is
never superseded by a configuration change:

* **H60 real adapter, real pinned Qwen3-VL-2B**: strict bare JSON validity
  0 → 17 of 48; geometric grounding **0 of 24** on both public DEV sets;
  real GUI drag task success **0 of 48**.
* **H46 real inference**: **0 of 8** correct on the original eight public
  first-party DEV photos (8 of 8 valid JSON, 3 wrong unique, 5 unbound).
* **H47 paired prompt study**: removing the numeric example did not improve
  the DEV model.
* **H36**: a full 64-of-64 real inference run on a free Kaggle T4 that
  produced strict JSON and no useful grounding.

A longer corpus, a better supervision weight and a warmer prompt are
hypotheses until a matched measurement says otherwise.
