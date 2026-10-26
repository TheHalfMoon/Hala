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

## H137 final one-shot independent MIT-library transfer evidence — 2026-10-26

Qualified source: **36/36** unique original 960x720 screenshots from *actual*
offline upstream **noUiSlider 15.8.1, Split.js 1.6.5 and SortableJS 1.15.6**
browser runtimes, not the H136 self-authored drag implementation. Each
third-party library receives an owner-authored mini-workspace and visible
instructions; **all 36 source-oracle pointer drags actually changed library
state in Chromium**, before any model inference. Exactly 6 Arabic plus 6
English cases per library. Manifest source SHA256
`c9762ca1a1ed4251cf42caba07199e4bbf5ed137a063b68619f79f00085fa324`.
No screenshot SHA reuse from H134/H135/H136, owner verified.

One private FREE Kaggle T4 run completed **144 actual pretrained and frozen
H136 adapter START/END neural calls**, **0 optimizer steps**, both 2B arms
loaded without modifying any weights. The H136 experimental adapter remains
at SHA256
`183680c3224b024c546dbbd1428e7a05ffa1491d57c117aec350a11e04cba1a7`.
Hala flagship 35B weights untouched. Owner Windows independently downloaded
the raw full responses, validated model arm/case/language/library role,
re-parsed every native coordinate and bbox hit, checked source/third-party
vendor byte SHAs, and executed **72 actual model-predicted mouse drags**
with no fallback to DOM or gold coordinates.

| Real MIT engine + instruction | Base external 2B | H136 64-step LoRA |
| --- | ---: | ---: |
| noUiSlider EN, 6 tasks | 5 | **6** |
| noUiSlider AR, 6 tasks | 5 | 5 |
| Split.js EN, 6 tasks | 0 | **1** |
| Split.js AR, 6 tasks | 0 | **1** |
| SortableJS EN, 6 tasks | 0 | **4** |
| SortableJS AR, 6 tasks | 0 | 0 |
| **All real browser full-task successes** | **10/36** | **17/36** |

Exactly **7 trained-adapter-only and 0 pretrained-only** paired browser
successes, two-sided exact discordance p=**0.015625** on this frozen
36-task owner-designed, foreign-library-implemented suite.

**Decision:** This provides *positive evidence of transfer to three actual
unseen JavaScript UI interaction engines* but is **not** proof of
generalization to untouched production applications. The most urgent
weaknesses are Split.js 2/12 after training and SortableJS Arabic 0/6.
It is NOT an updated 35B Hala model. These specific evaluation images are
now exposed: never train, retune prompts or select checkpoints from them.
Do not claim release readiness.

Next H138 should evaluate frozen base and H136 adapter against truly
independent upstream MIT *complete app repositories*, with their source
unchanged and pinned to exact Git SHAs, using reproducible owner-private
application local builds, real DnD persistence/event-success and bilingual
task instructions. Record both original language and controlled Arabic
paraphrase separately, without allowing oracle layout/coordinate info into
the prompt. Do not use H137 images to guide training.

Source, raw results, upstream JS binaries and trained adapter stay PRIVATE.
No deployment or publish, no paid compute, no merge/release.
