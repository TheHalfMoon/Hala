# H46 — frozen real Qwen3-VL-2B first-frame bilingual public DEV experiment

**Status 2026-10-20:** source implementation and Kaggle notebook builder; no actual H46 model result exists until a new provider Batch run terminates, raw receipts are retained, and the independent offline evaluator reviews every case. A synthetic unittest score is NOT a neural performance result.

Source base [H45 PR #29](https://github.com/TheHalfMoon/Hala/pull/29), exact signed `d23a5e2cae00e4b5e78f72c7d2883b12d32b8c3e`. [H44 PR #28](https://github.com/TheHalfMoon/Hala/pull/28) supplies immutable real Edge Chromium first-view captures. **Do not touch or rerun failed H41 H23 CUDA; its frozen font mismatch is an unrelated historical negative result and H23 CUDA is still unqualified.**

## Scientific question and preregistration

Can **the original pinned Qwen3-VL-2B-Instruct FP16** (`Qwen/Qwen3-VL-2B-Instruct`, exact HF revision `89644892e4d85e24eaac8bacfd4f463576704203`) correctly choose the intended first UI control from PNG screenshots and the original bilingual task intent without seeing model-evaluator ground truth?

- **Exactly eight previously authored public DEV images** from H44: `harbor_inventory` four (two EN/two AR) and `summit_scheduling` four (two EN/two AR). Sixteen authoring cases and H45 follow-up action screenshots are excluded. This is **not a sealed or independent TEST set**; all scenes remain first-party synthetic, and tasks can be visible in public repo.
- Original Qwen3VL input format and **unchanged H34/H37 revised prompt** `PROMPT`; 960 x 540 screenshots only + task `goal`, deterministic `do_sample=False`, `num_beams=1`, `max_new_tokens=96`, one original model attempt/case, bounded response <= 2048 bytes, zero retries. No HTML, DOM `data-ref`, original golden actions, scenario state labels, category hints, tool integrations or evaluator JSON passed to model. Runner code touches only `model_inputs.json` + eight PNGs; repository source clone contains evaluator files for later **separate** audit and is not filesystem-level sandbox isolation, but model has no tooling or filesystem API.
- Locked exact runtime `torch 2.11.0+cu128`, `transformers 5.16.1`, `accelerate 1.14.0`, `safetensors 0.8.0`, `Pillow 12.3.0` (historical H34/H37 model baseline, distinct from H23 Pillow 11.3.0). Original licensed pinned weights, `trust_remote_code=False`, safetensors, T4 observed by Kaggle and Python, explicit free noncommercial research-only environment. If versions, actual allocated free T4, network, model loader, source SHA or quota don't match, **stop and retain an honest incomplete run**; never silently pip-install different versions, substitute fonts, change decoder, retry/re-run, escalate to paid hardware or import another benchmark corpus.
- Captures all eight raw model outputs (or explicit per-case error) including unbound/off-target/abstentions and incomplete/missing attempts. Does not issue any UI click. An owned worker process must be killed/reaped on bounded timeout; journal fsync after each attempt; returns `model_qualified=false` whether it succeeds or fails. The Batch parent has an independent outer timeout; no GPU allocation is guaranteed by an API push.
- Real reviewer reads the **original raw SHA256** separately from provider output, checks source, prompt, input and evaluator hashes against precommitted records, rejects row swaps, duplicates, extra/missing records and metadata inflation, and scores normalized coordinates against the **complete observed set** of DOM buttons/controls, requiring exactly one intended target. Reports parse, target hit, uniquely wrong target, unbound, abstain and failure by AR/EN with unconditional denominator 8. Missing cases remain failures.
- This cannot be compared directly to H37 **32/64** because its H37 images/contexts/tasks differ; it is **not** a Holo4 benchmark, real user application, long-horizon model task, OSWorld 2.1 or ScreenSpot-Pro. No competitor win claim. Do not use public DEV for final blind benchmark or contaminate future sealed tests.

## Frozen data and model identity

| Input or evaluator | SHA256 |
|---|---|
| H44 model-only manifest | `f432747051c5fc2b6741db1a8c9b4b58f04d0dd7248c77546baa191bcdc29193` |
| H44 evaluator-only DOM first-step oracle (worker never reads) | `974b6ce6a51d272437e67471c10ee07f03b9303427bac9ec66e0391fe254ee22` |
| Actual Edge capture metadata | `ba87f75f412ba235903ec72e119fc8fb021455a646f8a0c9becb5adc455cd14c` |

Source file `h46/experiment.py` contains admission, worker, source-blob verification and **separate independent replayer**. Notebook generator `tools/build_h46_kaggle.py` creates a private, one-time source-checked Kaggle artifact at `abdulazizshehri/hala-h46-8-dev-qwen3vl-pixel-20261020`, with opt-in and no submission by default. It contains no evaluator controls/answers; model sees only screenshot bytes and task instruction. The notebook independently checks real Kaggle T4 before checkout and verifies every pinned source file after a clean clone of the exact signed branch commit.

## Launch procedure and remaining gates

1. Pass full source test suite, `python -m unittest tests.test_h46_real_model_protocol -v`, Python compile and real Graft **wiring** check; do not claim Graft semantic/deep if not run.
2. Create signed git commit, publish stacked **Draft PR** above H45, verify exact-head CI Python 3.11/3.12; do not merge automatically or claim independent code security review that didn't execute.
3. On a clean detached worktree of the exact published SHA, run `python -m tools.build_h46_kaggle --root . --source-commit <published SHA> --output <unique temporary directory> --research-allowed`, verify notebook SHA, private GPU metadata and Python cell syntax; submit **once**, only when authenticated Kaggle free GPU quota is verified and no existing kernel version exists. This is a separate new study, **not an H41 retry**.
4. If terminal, retain `raw_attempts.jsonl`, `run_metadata.json`, parent output, exact raw provider status/logs and every failure. Run `h46.experiment.replay(output, independently_calculated_raw_sha256)` offline from unchanged first-party source; report **8-case unconditional AR/EN DEV results**, and no unearned qualification. All evidence belongs in signed GitHub Hala repository; clean only verified byte-identical temporary copies after archiving.
5. Next genuinely model-improving experiment should collect **new training-only diverse workflows** (not benchmark trajectories), fine-tune legally/technically admissible small weights, then compare the model/adapter on identical independent tasks against this frozen baseline. Introduce sealed externally adjudicated Arabic/English and real tasks before any claim of competitor leadership.

**No rights are asserted over third-party OSWorld, Holo4 test traces, or external application data. No cost incurred by this source milestone.**

## Measured source-only Windows qualification — 2026-10-20

- H46 focused **4/4 test cases passed**, with simulated fake raw coordinates **strictly for evaluator parser tests**; no generated fake values are neural model results. `compileall` PASS.
- Complete serial Hala test suite **562 tests run / 560 passed / 2 skipped**, exit 0, in **133.025 seconds**. Original `reviews/h46-full-suite-20261020.log` SHA256 `03e1160842b459f063fbbc9e9451d3b01ae21d35c29305d799f2f4620e567623`. Does not qualify actual GPU/model.
- Genuine Graft `graft build .` and `graft check .` **wiring-only 1,757 nodes / 4,750 edges / 211 parsed files PASS, in sync**. **Deep meaning tier 0%; not run**.
- Independent code reviewers D1, Alibaba, PStack, e2e, Graft deep meaning **not claimed**. Host exact signed CI Python 3.11/3.12 still needs independent provider check after push.
- No paid computation, additional H41 failed notebook run, model weights or actual GPU allocation, native/real UI interactions in this source milestone; original `.ignore` and concurrent H41 evidence work unaffected.
