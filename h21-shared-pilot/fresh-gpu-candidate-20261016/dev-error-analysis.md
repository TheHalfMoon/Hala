# Retained candidate: DEV-only error analysis

The gain is weak: candidate completes 1/8 DEV tasks (English 1/4, Arabic 0/4), versus captured Kaggle base DEV 0/8. Plans improve to 8/8, but correct initial decomposition does not establish successful state-dependent execution. This analysis uses only the eight DEV episodes and the independent 54-action geometry replay. It does not analyze TEST failures or the ten standalone test-labelled probes for remediation. No training, data, prompt, parser, or production source changes were made.

| Language/action | Attempts | Action errors | DEV explanation |
|---|---:|---:|---|
| English type | 4 | 0 | All field targets and requested text correct |
| English drag | 16 | 12 | Twelve repeated drags after destination already reached |
| English click | 2 | 0 | One saves; one triggers the authored transient failure |
| Arabic type | 18 | 16 | Sixteen field-point misses in the reading-shelf pair, despite correct text |
| Arabic drag | 0 | 0 | No card-move proposal occurs |
| Arabic click | 14 | 14 | Seven premature save target hits, seven save misses also premature |
| Total | 54 | 42 | All 42 error actions leave state unchanged |

The categories above are disjoint for counting but some actions have multiple causes. Twelve English repeated drags start and end inside the destination region after the card was already moved; their starts also miss the original item box and their moves violate the source-location prerequisite. Seven Arabic premature-save actions hit the save rectangle correctly but fail because the card is still in its source location. The other seven Arabic clicks miss the save target and likewise omit the required move. Thus coordinate accuracy alone would not resolve all failures. All 22 type proposals carry the exact requested text, so DEV does not indicate a text-content problem. The sixteen Arabic field misses repeatedly use a point around the card's vertical level rather than the field, and make no field update.

Mind contributes 33 semantic errors out of 59 calls: 28 are schema-valid `move_card` decisions inappropriate to current state, and five are invalid tool/subgoal combinations. Fourteen valid wrong decisions occur in the English episodes: two after the reading-shelf move and twelve while already-moved pet-journal episodes repeat dragging. Fourteen occur in Arabic reading-shelf episodes, where Mind requests moving although the failed type left the field unchanged. In Arabic pet-journal episodes Mind's 16/16 semantic decisions are correct, yet Type skips moving and repeatedly clicks; correct Mind scoring therefore does not imply the visual action follows the intended next state transition. The runner supplies the public goal/state to Type and does not enforce the Mind subgoal as an action precondition.

The five `model_errors` are specifically five Mind `ValueError`s in the English transient reading-shelf episode. Every one returns `{"kind":"next","tool":"visual","subgoal":"inspect_failure"}`; `inspect_failure` belongs to `observe`, so the strict protocol rejects the pair before another visual call. They are not five GPU failures, timeouts, bad visual JSON proposals, or five additional errors to add to the 33 semantic errors. These five are included in the semantic-error total. The episode correctly reaches a transient save failure but never observes/retries, leaving zero recoveries. The one authored environment error is separate from the 42 action errors and the five model errors.

The initial plans reproduce the required replace/move/save/verify sequence in all eight episodes. Subsequent observations expose the missing skills: recognizing that a proposed action did not change the field, recognizing that a successful move is already complete, executing a drag when the next semantic operation requires it, and routing a transient failure through observation before retry. These observations explain why a strong plan score coexists with weak completion; they do not prove a particular training cause or that any proposed remedy will work.

If a later, separately authorized development phase is pursued, DEV supports these data priorities:

- Add varied Arabic mirrored-layout field and save localization examples, including correctly transcribing text while selecting the correct target. Vary geometry and labels so coordinates cannot be learned as a fixed template.
- Add state-transition examples pairing before/after observations with the next required operation: failed type still needs replacement, completed move needs saving, and unchanged state needs reassessment. Include examples where the repeated previous action is explicitly inappropriate.
- Add visual next-action examples that distinguish moving a card from clicking save after a field edit, particularly Arabic states where Mind already requests the correct move. Treat semantic choice and visual execution as separately measured skills.
- Add transient-failure examples with the valid `observe/inspect_failure` pair followed by an actual retry and persisted-state check. Preserve strict rejection of invalid pairs rather than interpreting them as successful recovery.

Any such work should use fresh training examples and DEV validation only, with a newly sealed future holdout; the already exposed standalone probes must not supply training targets or remediation feedback. No new pilot or automatic retraining was run. The eight authored DEV episodes cover two small families with paired variants and shared templates. Repeated errors are correlated, so the counts are descriptive and do not establish general Arabic proficiency, a reliable improvement rate, or a general release claim.

Machine counts and 42 independently classified action-error records: `work/h21-candidate-dev-error-analysis.json`. Reproducer: `work/h21_candidate_dev_error_analysis.py`.
