# H25 — prospective bilingual semantic decision dataset

**Status: SOURCE-ONLY candidate, not model-trained, visually qualified, externally semantically reviewed, or tested on genuine Orcel device execution.** Prepared 2026-10-18 on a fresh isolated Hala worktree stacked on draft H24 PR #5. Historical frozen H19/H22/H24 dataset/test branches remain unchanged.

## Reason for creating this new source

Actual independent byte-verified H24 DEV V1 and V2 both measured **zero accepted strict HalaMind proposals out of twenty** on the same synthetic family-dependent tasks. V1 had 6/20 diagnostically correct raw tool names (invalid proposals); V2 collapsed into `kind=abstain` for all 20 raw responses. Those DEV cases are now *adaptive development feedback*, not a valid novel holdout. Repeating small-prompt variants on them could inflate metrics without changing agent quality.

H25 starts with **84 new, first-party original authored text-only examples**, not a reuse of H24/H22 cases or raw model replies. Source `h25/corpus.py` authors distinct domain language and state-dependent decision problems across six independently named domains: instrument calibration, community food distribution, museum object registration, seminar scheduling, accessibility auditing and conservation habitat survey.

- **Authoring:** 42 cases in three new families (21 Arabic/21 English). Prospective content development only, *not* an authorized training set; rights flag `commercial_training_clearance=review_required`.
- **DEV:** 42 cases in three other new families (21 Arabic/21 English), unmeasured so far. They are **not TEST**, and cannot be called an independently protected test set once their contents appear in public Git.
- Per family/language: seven separately written scenarios covering actionable UI targeting, stale observation, missing required value, hostile webpage instruction, explicitly destructive operating-system request, saved-state postcondition verification and contradictory signed evidence.
- Arabic text is verified at the Unicode codepoint level; no text-shaping or screenshot output is claimed. GUI control bounds are synthetic metadata, not proof of visual grounding, Orcel security permissions or a real GUI.
- Deterministic, auditable synthetic `route()` specifies a private expected `kind/tool/subgoal`. It handles destructive user requests by abstaining, untrusted page text by ignoring it and re-observing, stale frames by re-observing, missing values/conflicting evidence by clarification, and unverified saved state by requesting verification. **This policy is a synthetic evaluator, not a trusted device policy.**
- Unlike the H24 five-option prompt, H25 introduces `request_missing_parameter`, `check_persisted_postcondition`, `resolve_conflicting_evidence` outcomes. Those require separately versioned typed contract review and should not be silently passed to the current production or V2 five-pair vocabulary.

## Raw immutable candidate

| File | SHA256 |
| --- | --- |
| `h25/frozen/cases.json` | `b94942b1b1a870867214b87160a7aad06776f4be8d111dcfa111041f7cd5759b` |
| `h25/frozen/manifest.json` | `b4c27bba3209708d9892ddb6f4987027183a116d55421c07deca3e2d43470be3` |
| `h25/frozen/public-dev.jsonl` | `5379f3db559c304c037e971cdf6aa15020211f57c42e0bb30fb75a643f947e57` |

The DEV projection exports only `source_ref` (case tracking, never prompt input) and `model_input` containing goal, locale and public observation. An eventual runner must send **only `model_input`** and audit the reference separately. All private oracle labels, rights and split metadata are excluded from public LLM input; no training export was generated.

Regenerate from a clean checkout into a new directory:

```bash
python -m h25.corpus --output NEW_NONEXISTENT_DIRECTORY
python -m unittest tests.test_h25_corpus tests.test_h25_frozen -v
```

The source regenerates and validates exact bytes, Unicode goals, cross-family partitions, synthetic coordinates, safety routing, rights/labels and all three immutable SHA256s. The write command will not overwrite an existing directory.

## Gates not achieved

- **Independent Arabic/English semantics audit:** Original authored sentences are not proof of expert-quality Arabic interpretation; language-native review and open-source data/legal verification are needed before training use. Neither copyright/rights metadata nor synthetic source alone guarantees commercial redistribution rights.
- **No screenshots, no RAQM/RTL shaping or HalaType Type supervision.** No trust in synthetic coordinate layouts beyond source-based tests.
- **No actual GPU inference, HalaMind improvement, LoRA, native Orcel integration, model selection, measured cost/latency, H23 CUDA cancellation or production approval.**
- **No H25 TEST claim.** Future scientific evaluation requires truly new task families, ideally unpublished/protected from model selection, and representative multi-step causal GUI outcomes.
- **Independent fresh-context PStack/security and Alibaba model-based review still separate.** Genuine Jev/Graft and unit testing are engineering evidence, not human trust or real model/task success.
- **No edits to Orcel**, its policies, its approvals or executor. All Hala outputs remain untrusted typed proposals; Orcel alone authorizes actions and verifies outcomes.

Next: independent semantic review of the 84 examples, family-level factual task checks, qualify a new held-out multi-step simulation and Arabic visual renderer; only then propose one justified, free-compute HalaMind baseline and an independent evaluation protocol. No adaptive recycling of H24 DEV or H22 TEST.
