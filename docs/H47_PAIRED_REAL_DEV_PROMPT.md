# H47 — pre-registered same-T4 paired prompt intervention, genuine model DEV only

**2026-10-20 — source protocol; no H47 GPU results yet.** Actual H46 [provider-negative result and independent 0/8 replay](https://github.com/TheHalfMoon/Hala/pull/31) is the immutable signed parent `34756b41b8219200b9a14af3f5e717435e83e443`. H46 itself had 8/8 parseable model raw points, 0/8 intended control, 3/8 uniquely wrong, 5/8 unbound; 0/4 Arabic, 0/4 English. Eight actual first-party synthetic Edge DEV tasks are not a blind/third-party benchmark.

## Hypothesis, intervention, control and fixed dataset

The original H34/H37 `NUMERIC_EXAMPLE` prompt includes a JSON output coordinate example `[0.25,0.75]`. In H46 original raw output it occurred twice, suggesting (not proving) example-position anchoring. H36 already defines an established, carefully minimal alternate prompt `NO_NUMERIC_EXAMPLE`, replacing the example with a description of two normalized point numbers but otherwise retaining response format and task. **H47 directly reuses H36 source** and does not create a new unrestricted prompt strategy.

An actual private Kaggle experiment compares **both** prompt arms on each of the **same eight frozen H44 genuine Chromium first-view images** (4 English, 4 Arabic), under the **same** pinned Qwen3-VL-2B Instruct FP16 revision `89644892e4d85e24eaac8bacfd4f463576704203`, original processor, one genuinely allocated free T4, pinned exact H37 package versions, one model load, no sampling and exactly `max_new_tokens=96`. Every case is run under both arms, with **4 cases original-first and 4 no-example-first**, alternating by case index. Sixteen trials, zero retries, immutable source/PNG hashes and per-arm prompt digests, separate independently checked paired journal. This is a **public DEV diagnostic** only.

**No answer leakage:** model sees only original PNG bytes and bilingual user goal plus assigned prompt. No DOM, expected button, normalized correct target, H44 evaluator JSON, privileged task records or any tool/OS execution. Model code running on Kaggle contains source files in the cloned repo and is not physically sandboxed away from evaluator files, but no model can browse files and the model-facing code never opens evaluator content. Independent CPU replay uses external authentic H44 observed-control rectangles, not success inferred from a generative statement.

## Strict outcomes and statistical caveats

Independent `h47.experiment.replay` requires actual raw SHA256, immutable exact source and original browser/evaluator hashes, fixed 16-record order, two prompt hashes, exact task/image/arm bindings and bounded error/raw schemas. It always counts **8 tasks per arm**, all 16 outcomes or missing attempts, with English/Arabic breakdown, correct unique target, wrong unique target, unbound, abstain, parsing and runtime failures. Pairwise outcomes count candidate-only wins, control-only wins, ties and incomplete paired attempts; missing runs cannot be quietly transformed into ties.

A numerical increase on these eight public tasks **does not establish generalization**, a winning model, significance, multilingual mastery or a production release. Repeated diagnostics on public DEV may bias future engineering decisions and must not be used as an independent TEST claim. Do not compare H47 to OSWorld/ScreenSpot/Holo4 leaderboards; do not infer causality or broad improvement from at most eight pairs.

## Execution and decision gate

1. Run CPU focused protocol/failure tests and complete Windows suite. Graft wiring-only build/check; do not claim deep semantics unless actually run. No D1/PStack/Alibaba result claimed without independent evidence.
2. Sign and publish exact H47 source on stacked Draft PR above H46 evidence, verify GitHub exact-head Python 3.11 and 3.12 CI.
3. Build **one private create-only** signed notebook `tools/build_h47_kaggle.py` from a detached clean commit, archive original notebook metadata and hashes on a separate evidence PR, verify Kaggle free GPU remaining and nonexistent kernel ID, then submit a **single** private version 1; never retry failing H46/H41 notebooks.
4. Wait for terminal status; retain original Kaggle metadata and `raw_attempts.jsonl` + external checksum and independent eight-pair replay in GitHub. Regardless of success, the next step toward a true model improvement must be new rights-cleared training-only diverse visual trajectories and a *separate independent heldout English/Arabic evaluation*, not optimizing repeatedly on this same public DEV pilot.
5. No paid services, no new native desktop actions and no production or benchmark claim before authentic independent performance evidence and required governance.

## Actually observed local H47 source gates — October 20

- Focused **5/5** CPU tests PASS (paired order and both real frozen prompts, fake-only parser regression, explicit missing pair handling, private notebook opt-in/source boundaries). One genuine bug in incomplete pair accounting was found/fixed before final tests. `compileall` PASS.
- Full serial Windows Hala test suite **567 tests / 565 passed / 2 skipped**, exit 0 in **100.748 seconds**. Retained raw transcript `reviews/h47-full-suite-20261020.log` SHA256 `7d96f55ec5c062aea2e8d484df211df93643be5f8bdf2e5abd4baa888839bb5f`.
- Genuine Graft **wiring-only** build/check PASS **1,780 nodes / 4,826 edges / 215 files**, graph consistent; **deep semantic meaning tier 0% / not run**.
- No actual H47 GPU/model run or claimed improvement yet, no independent PStack/D1/Alibaba code review, no paid services, no native UI actions, no alteration of H41 original frozen CUDA result or H46 immutable 0/8 raw evidence.
