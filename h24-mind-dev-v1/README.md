# Hala H24 — real, measured, unqualified HalaMind DEV baseline V1

Date: 2026-10-18 (Asia/Riyadh). Kaggle source: [private Saved Version 1](https://www.kaggle.com/code/abdulazizshehri/hala-h24-mind-dev-baseline-20261017).

## Scientific identity

- Kaggle worker reached `COMPLETE`, and downloaded `result.json` internal status is `measured_unqualified`.
- Authored H24 **DEV-only**: twenty original synthetic text-only cases, ten English and ten Arabic, across two distinct held-out application families. H24 **TEST was not used**. No HalaType images, native device actions, closed-loop task benchmark, original H22 TEST reuse, training or LoRA.
- Exact published evaluator/model source SHA: `f9bd2b011fb2f7dceed340e52d994a247888e833` (Draft PR #5).
- Base `Qwen/Qwen3-VL-2B-Instruct`, model revision `89644892e4d85e24eaac8bacfd4f463576704203`; reported Tesla T4, FP16, deterministic generation, no adapter.
- A separate CPU auditor `tools/verify_h24_dev.py` from the published Hala source independently verified the externally observed raw JSON byte SHA256, all 20 row identities and exact source/model pins, independently rescored every raw reply and verified the bilingual summary. Report `verified-h24-dev-v1.json` attests **byte/label consistency only**; it does not attest the physical GPU or establish general model quality.

## Observed metrics

| Outcome | English DEV | Arabic DEV | Total |
|---|---:|---:|---:|
| Raw outputs valid JSON syntax (post-hoc descriptive inspection) | 10/10 | 10/10 | 20/20 |
| Expected 3-key `kind/tool/subgoal` fields, ignoring strict subgoal grammar | 10/10 | 10/10 | 20/20 |
| **Executable typed-contract valid** (strict normalized `subgoal`) | 0/10 | 0/10 | **0/20** |
| Correct tool choice ignoring invalid typed-contract status (diagnostic only) | 3/10 | 3/10 | **6/20** |
| Inference exceptions | 0/10 | 0/10 | 0/20 |

Original published `result.json` has `summary.*.valid_json=0` because its `score()` definition uses **strict typed-proposal validation**, including the restricted normalized `subgoal`, not JSON syntax alone. **Do not reinterpret or overwrite its original metric.** The separate `diagnostic-v1.json` conservatively decomposes lexical JSON syntax, tool choice and typed contract to explain the failure. It does **not** promote a structurally invalid model reply into an executable action.

The model often returned a free-form instruction such as `Click on the 'Review ...' button` inside `subgoal`, which does not meet the strict identifier grammar. It also selected `visual` in **10/16 non-valid scenario cases** (stale frame, ambiguous destination, hostile page or unsafe user request), although it did **not** choose `visual` on any of the four synthetic unsafe-user-command cases. These are evaluator classifications on untrusted model suggestions; no operating-system action was executed or authorized. The numbers come from twenty dependent synthetic DEV prompts, so they cannot estimate real-world Arabic task success or safety.

## Exact archived raw bytes

| File | SHA256 |
| --- | --- |
| `result.json` | `a047f7652b1a954c745ba139431299602961cd017ad547253acea1b5fc24ef4f` |
| `verified-h24-dev-v1.json` | `0296045d6b0e0dd23e5ac60e05ff117fdb93608271e37d822c1719600a5df962` |
| `diagnostic-v1.json` | `69bf1ca95be709e3aaef70b4ebaa71c4bf3e1541b37c971f760c584ba2128eac` |

Immutable outcome: keep raw V1 unchanged, report both strict-contract and post-hoc diagnostic metrics transparently. Any DEV prompt revision or constrained decoding should use a separately defined prospective protocol and should **not** tune on H24 TEST cases. No model or precision is approved for deployment; no Orcel repo changes or H23 real lifecycle validation. Founder retains merge and release decisions.
