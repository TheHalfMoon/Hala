# H137 — first cross-library generalization evaluation

H136 was a separately trained 64-step rank8 language LoRA adapter on an
externally attributed pretrained GUI Assist Grounder Qwen3.5-2B.
It completed 15/24 heldout tasks built from the *same owner's custom
workstation generator*; its 35B flagship Hala counterpart remains unchanged.
Synthetic same-generator results do not establish cross-application skills.

## EVAL-ONLY independent interaction runtimes

These test cases run **real upstream MIT JavaScript UI engines** installed
in a private offline owner-local workspace, not H136's custom drag logic:

- **noUiSlider 15.8.1** (MIT), actual slider thumb and on-change value.
- **Split.js 1.6.5** (MIT), real split pane gutter and onDragEnd width.
- **SortableJS 1.15.6** (MIT), actual card transfer across sortable lists.

These are owner-authored mini-workspaces wrapping genuine third-party
interaction engines, **NOT untouched independent production applications**.
This matters: they test changes of the *drag implementation family*, not
full real web-app generalization.

36 freshly rendered original 960x720 Chromium screenshots were collected,
12 per library, 18 Arabic + 18 English, balanced 6 per language per library.
The true start bounding boxes refer to real library-rendered controls;
the destination element is visible in the screenshot. All 36 true
owner-reference drags actually changed the library's underlying state and
passed, proving the tasks were physically solvable before model inference.
Gold points and DOM were not part of the model prompt.

Screenshots, references, npm node_modules, vendor files and all weights
stay owner-private. Repo includes only the test harness, documented
package versions and replay tool; upstream license and attribution are
preserved in the private vendor installation.

## Frozen benchmark / no optimizing on test data

- One owner-private FREE Kaggle T4 execution, zero optimizer steps, exactly
  144 genuine model calls (36 × START/END × pretrained/H136 LoRA).
- Frozen upstream 2B `patrickballou/gui-assist-grounder` at
  `31632edaa20749c7f532868a42635f7240033106`.
- Frozen H136 experimental adapter SHA-256
  `183680c3224b024c546dbbd1428e7a05ffa1491d57c117aec350a11e04cba1a7`;
  Hala flagship 35B remains untouched.
- Original screenshot inputs at 960x720 with published Qwen processor,
  same predeclared H136 English role prompt around native AR/EN task.
- Owner independent source and native model JSON validator, then real
  Playwright mouse down/move/up to both neural predicted coordinates with
  no oracle fallback.
- Primary: actual library state/task success, stratified by library and
  instruction language. Secondary: independently reproducible START/END
  bounding-box hits and exact paired discordance test.
- Any results are EVAL-ONLY; do **not** tune, re-train or change prompts
  on these 36 cases or repurpose them as future training data.
- If this fails, diagnose the mismatch and build a substantively new
  future independent benchmark; do not reuse this exposed test for
  promotion claims.

Still pending beyond H137: human-reviewed real screenshots spanning full,
untouched applications and Arabic/English workflows, the independent H127
genuine H71 semantic gate, long-horizon computer-use tasks, security and
recovery qualification. Never infer readiness from toy cross-library success.

No paid cloud, deployment, public weights, public datasets, merge or release.
