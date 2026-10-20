# H36 — Actual private Kaggle paired prompt trial (2026-10-20)

**Immutable source commit:** `ae3f684580c5cd3fba63442738b945de88ad1085`. H36 research notebook on Kaggle: https://www.kaggle.com/code/abdulazizshehri/hala-h36-paired-prompt-dev-20261020 (private). **Pinned source branch:** Draft PR #17 remains fixed at that exact commit; this evidence is added on a separate stacked Draft branch, without moving the source branch. Future notebook reproducibility requires that source SHA to remain available. No model adoption or release.

## Source/environment and raw receipts

- 64/64 actual trial records retained in `raw_attempts.jsonl`, source SHA in `run_metadata.json`, independently computed raw SHA256 **`761cceef06f588a67286093580e6f1a1cfe3d3d7605c2b650bf3c9e7c815355e`** (equals Kaggle console output).
- Same model loaded for both arms: `Qwen/Qwen3-VL-2B-Instruct`, exact revision `89644892e4d85e24eaac8bacfd4f463576704203`. Observed Kaggle Linux Batch T4 devices (2 visible; inference on `cuda:0`), exact observed pins `torch 2.11.0+cu128`, `transformers 5.16.1`, `accelerate 1.14.0`, `safetensors 0.8.0`, `Pillow 12.3.0`, RAQM 0.10.5. GPU metadata is **observation, not cryptographic attestation**.
- `run_metadata.json`: state completed, worker exit code 0, no deadline overrun, 64 attempted, approx 136.57 worker-wall seconds. The Kaggle Batch notebook completed. The code executes **no native/UI actions**.
- All **32 original real rendered H34 PNGs** are retained with `image_manifest.json` (per-image SHA256, pinned font/license/renderer) and `cases.sha256`. Independent comparison confirms **32/32 PNG SHA256 values match H34's prior revised-prompt Kaggle run**, and **32/32 byte-exact control-arm model outputs match the previous H34 raw model responses**. This is strong deterministic control replication on exposed DEV, not new generalization.
- `plan.sha256`: **`39c11abf83cfb997022378d0a3fa516a229ac0209b38de314e8786535671c792`**, 32 cases × 2 within-VM interleaved arms.
- `audit.json`: **independent Windows CPU strict replay** of raw digest, pinned image bytes/PNG CRC, trial order, source prompt hashes and H34 scorer. SHA256 **`08d18dceb890cff7ec18c07b7ea6b4291414ee9ffcd7bac2b33c2cfb2fec251f`**.
- `run.log` is the Kaggle CLI downloaded console receipt, SHA256 `1a9b19e00b21229477e9ffa53156ab94d977ced23b1bed5fbbf17b5ba2334a17` (credential pattern check: zero matches); it is not evidence of independent approval.

## Measured real model DEV results

| Metric | Historic numeric sample prompt | New no-numeric-sample prompt |
|---|---:|---:|
| Attempted | 32/32 | 32/32 |
| Strict JSON parse valid | 32/32 | **22/32** |
| Intended target hit (unconditional) | **8/32** | **6/32** |
| Wrong visible-control hit | 2/32 | **5/32** |
| Valid unique control binding | 10/32 | 11/32 |
| Abstentions / runtime errors | 0 / 0 | 0 / 0 |
| English intended hits | 3/16 | 2/16 |
| Arabic intended hits | 5/16 | 4/16 |

Paired tally: 3 cases both prompts correct, 5 only the historic prompt correct, 3 only the no-numeric prompt correct, 21 neither correct. **Removing the example did not improve visual target grounding in this DEV experiment; it reduced parser validity and correct-target hits.** No hypothesis is established as a general causal effect; both arms share the already-exposed H34 DEV cases, not independent blind TEST.

### Exploratory failure forensics (post-hoc; do not tune the frozen evaluator)

The historic numeric-sample arm repeated normalized **y=0.75 on 8/32** and each point was off the displayed controls, matching prior H35 forensic finding. All **10** `no_numeric_example` malformed responses contained **duplicate JSON `point` keys**, rather than Markdown fences. This is a new observed schema-fidelity failure mode from the *same exposed DEV*. Continue to reject duplicates; do not silently parse the first/last key or snap points onto controls. A genuinely prospective separate study could compare one-key constrained formatting with no concrete sample coordinates, then evaluate new disjoint GUI families and real RTL screenshots.

## Independent source and evidence gates

- Frozen H36 source commit `ae3f684` exact-head [GitHub Actions](https://github.com/TheHalfMoon/Hala/actions/runs/39364273929) **SUCCESS** on Python 3.11.14 and 3.12.14.
- Updated **evidence branch** actual Windows full Hala regression **495 tests run: 493 passed / 2 skipped**; H36 focused **8/8 passed**, including strict raw/oracle agreement and byte-exact H34 control replication; compileall PASS.
- Genuine Graft **wiring-only** graph build/check PASS at **1,544 nodes / 4,084 edges** (meaning/deep tier 0%, not performed). Actual independent model-driven D1/PStack/Alibaba review is not claimed.
- Replayed `h36.audit` again under a separate clean detached source worktree at exactly `ae3f684` after evidence download; 64/64 records and both paired arm results agreed.

## Replay without network, GPU or model

```powershell
python -m h36.audit --output reviews/h36-kaggle-run-20261020 --expected-raw-sha256 761cceef06f588a67286093580e6f1a1cfe3d3d7605c2b650bf3c9e7c815355e
python -m unittest tests.test_h36_paired_prompt -v
```

No local model weight downloads, paid compute or subscriptions. The Kaggle GPU quota was read from the official CLI immediately before launch (29.03 hours free remaining then), but resource use is observed not charged/billing-proof. Private research and zero-cost-only rules applied.

**Do not claim this experiment qualifies HalaType, HalaMind, H23 CUDA in-flight cancellation or native computer-use.** H23 operational source has stricter historical same-VM/package gates and was NOT run under H36. Independent Arabic expert adjudication, new disjoint visual TEST, prompt-injection resistance on real captured interfaces, production host safety and release approvals remain open.
