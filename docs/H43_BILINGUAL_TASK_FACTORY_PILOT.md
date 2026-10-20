# H43 — new bilingual first-party app-state task pilot with independent postcondition checker

**Scope:** synthetic CPU-only task factory, **NOT real GUI**, not new model training or neural inference, no benchmark result, no native OS actions. H43 branches from [H42 PR #26](https://github.com/TheHalfMoon/Hala/pull/26) signed head `70212c34b07016db35b5c1b450ae1113e96b907b`. H41's ongoing private Kaggle CUDA Batch experiment is separate and must not be duplicated.

## Why H43 is not another safety-only wrapper

The demonstrated Hala bottleneck is original neural target selection: actual H37 Qwen3-VL-2B predicted correct controls only **32/64** on first-party authored browser screenshots (Arabic 14/32; English 18/32; grid 3/16). H38/H39 model-free gates can refuse bad proposals, but don't teach the model to choose correctly. Before costly adaptation we need rights-cleared, reproducibly verified supervised **action/state** examples and honest independent holdouts.

H43 contributes an authored *pilot dataset* with executable, explicit final-state verifier and negative near-misses that reject spurious task completion. These tasks are **not GUI pixels**; this pilot alone cannot improve or evaluate HalaType visual grounding. The next work should use the checker to back a genuinely rendered first-party app UI, original screenshots and controls, then train/evaluate on separately qualified data and true external tasks.

## Actual reproducible source and split

New `h43/factory.py` authors **24 original CC0-1.0 fictional tasks** (12 English, 12 Arabic). Six fictional app contexts (each four cases):

| Split | Fictional family | Layout metadata |
|---|---|---|
| Authoring only | Atlas supplies | List |
| Authoring only | Cedar room planner | Grid |
| Authoring only | Orbit lending | Table |
| Authoring only | Delta shift roster | Sidebar |
| Public DEV only | Harbor inventory | Grid |
| Public DEV only | Summit scheduling | Table |

**16 authoring-labelled records** are exportable with `training_projection` (8 EN, 8 AR). **8 public DEV cases** are not exportable through the importer (4 EN, 4 AR). Complete canonical source identity and permissions are reauthored and compared exactly; reject relabeling DEV as authoring, changed examples/labels, personal or benchmark data, injected image refs and provenance changes. All inputs are original fake item IDs, none are user data, screenshots or third-party benchmark trajectories.

The split is **family-disjoint**, but the six contexts share a near-identical *three-step state machine* (select intended item, fill bounded quantity, confirm without modifying another item). That does **not** prove generalization to unseen action schemas or real applications. The public DEV data are visible in Git and **NOT independent/blind TEST**. No sealed task set was generated and no copied Holo4 benchmark trajectory was used for training.

## Independently adjudicated final state

`h43/engine.py` computes task state from each simulated action, separate from `golden_actions`, and judges exact allowed final state plus absence of unrelated mutation. It does not simply compare the action sequence to the golden script; a different harmless sequence that reaches the same state can pass. The checker tests:

- **24/24 golden trajectories** attain target state (positive);
- **24/24 untouched initial states** do not;
- **6 realistic wrong plans per case, 144/144** refused: choose wrong item, wrong amount, omit confirmation, confirm too early, modify unrelated item after success, and change selection to decoy before confirming;
- **168/168 combined negative cases** refused; all 24 cases audited, not success-only filtering.

Action schema strictly bounds ref/kind/value and max actions, rejecting malformed, wrong-role, noninteger/bool quantity, extra fields and unbounded attempts. A mock/app-local log is *not* an authenticated external truth: a malicious caller can fabricate state, and no browser/device executor is wired. This is a deterministic first-party **synthetic oracle rehearsal**, not proof of security or ability on actual apps.

## Reproducible retained source artifacts

| File | SHA256 |
|---|---|
| `h43/data/pilot_manifest.json` | `fbcf271e8c65637f2a69e7dbc5966d7ffdad29c7061d93684bf9a705e7f4c2c8` |
| `h43/data/pilot_verification.json` | `0c3d4562bd19bb665bfa45a21c442fcef9df39104afb3960260a59abc4f7d1b0` |
| `h43/data/authoring_projection.json` | `5f0867b124b1db0098e9a4936802776fb3afa4dd4cd18e952ed280e61bc27ee4` |

`tests/test_h43_task_factory.py` validates exact bytes/hash, balanced original languages, authoring/DEV split, all golden and negative replays, alternate valid path accepted, unauthorized side-effects refused, tamper/rights injection, honest projection and malformed actions. These outputs live in Git; no model weights, external data or hidden personal content were copied.

## Concrete next steps toward competitive model

1. **H44:** Render these **original** app states as actual pixels in headless Chromium with canonical screenshot+DOM digest, Arabic RAQM/RTL/font preflight, four layouts and independent target geometry. Ensure authentic observed window/control pixels, no privileged evaluator fields are model inputs. Record the screenshots and true source/license; add new, harder workflows rather than copying the identical 3-step state machine across apps.
2. **H45:** Create deliberately different training-only multi-step families (modal dialogs, editable tables, scrolling, asynchronous errors, recovery) with rights provenance; reserve completely separate, truly blinded independent cross-app tasks outside the training repo.
3. **H46:** Compare frozen original Qwen3-VL-2B, an eligible small open model and approved fine-tuned adapter on the **same untouched DEV and independent TEST** with exact checkpoint, model revision, GPU/hardware, prompt, budget and all failures. Don't train until memory, source rights and independent split are qualified.
4. **H47:** Only claim improved GUI grounding if model itself improves, not only fewer permitted clicks; do real long-horizon OSWorld 2.1/ScreenSpot-Pro with official harness, safe host permissions, Arabic expert adjudication and reported cost/latency.

**No H43 model quality gain or competitive benchmark score is claimed.** H41 historical CUDA qualification and original Qwen3-VL-2B 32/64 model grounder remain open. Founder release/production approval not granted.

## Actually observed local engineering gates — Windows, October 20 2026

- **8/8 H43-focused CPU tests PASS**, with per-case checks covering all **24 golden** and **168 negative** simulated state transitions, family isolation/rights and malformed action/label refusals.
- Complete Hala `python -m unittest discover -s tests -q`: **550 tests run / 548 passed / 2 skipped**, exit 0, **318.400 seconds**. Genuine original transcript `reviews/h43-full-suite-20261020.log` SHA256 `8f07372994b8497c05abd09c2a29a85ef16addf13557a7e14df1fd48e0438d91`. `compileall` PASS.
- Genuine Graft **wiring-only** build/check PASS **1,704 nodes / 4,590 edges / 199 files**, source graph in sync. **Deep meaning layer 0% / not built**.
- No LLM-based independent security reviews from D1, PStack, Alibaba OpenCodeReview or Jev are claimed. Hosted Python 3.11/3.12 CI must separately pass exact new signed GitHub head.
- No paid inference, cloud GPU, native application actions, genuine browser pixels, model checkpoint, private data or user account accessed. H41 queued Kaggle research GPU experiment remained a separate, unchanged submission.
