# H138 — frozen model on two untouched upstream MIT apps

H137 external 2B pretrained model versus frozen 64-step H136 LoRA achieved
10/36 → 17/36 real mouse task successes (paired p=0.015625) on three
genuine UI libraries, but the surrounding app shells were author-written.

H138 qualifies transfer on **unchanged upstream complete React apps**:
- MSabriBoy/Kanban_Task_Board React dnd-kit, MIT © 2026 MSabriBoy,
  SHA `553d4e92be3e86e0deb7045f4968152868399eae`.
- 9bzero/kanban-board React native HTML5 DnD, MIT © 2026 9bzero,
  SHA `83413e1f5fab4690bb46b23201a3438a427ef265`.

Both pinned Git checkouts are owner-private and built locally. App source
is unchanged. The evaluation runner seeds browser localStorage with
owner-authored example cards without modifying application source.
8 newly SHA-qualified Chromium 960x720 screenshots (four per app,
2 English and 2 Arabic external instructions) passed genuine mouse
drag reference tests with **actual persisted destination state**.

H138 frozen source SHA:
`203e5976131e685e296c4cf71edffe3c4332d9103fb7d508b041a850ce07e7c7`.
Evaluation-only. Do not train, tune prompts or select future adapters
based on these exposed benchmark images. Actual original apps are
more realistic than H137 but still small and not production deployments.

The next stage is a separate private FREE T4 32-inference no-optimizer
baseline-vs-verified-LoRA run, followed by exact model-proposed mouse
replay inside both untouched applications, with persisted state checked.
Frozen external base and H136 research adapter only; flagship Hala 35B
never updated. All source, data and adapter assets stay private. No
paid cloud, publication, deployment or merge. H127 real H71 human
source-identifiability gate and broader real user workflow qualification
remain open.

## H138 final frozen external full-app results — 2026-10-26

The exact original two MIT app revisions built unmodified on owner Windows:
`MSabriBoy/Kanban_Task_Board` SHA
`553d4e92be3e86e0deb7045f4968152868399eae` (React dnd-kit);
`9bzero/kanban-board` SHA
`83413e1f5fab4690bb46b23201a3438a427ef265` (React HTML5 DnD).
8/8 authentic source-reference mouse drags moved target cards between
columns AND persisted the desired state. 4 scenarios/app, 2 English and
2 Arabic external instructions per app; full unmodified UI source stayed
private. Heldout screenshot manifest SHA256
`203e5976131e685e296c4cf71edffe3c4332d9103fb7d508b041a850ce07e7c7`.

One private FREE Kaggle T4 eval completed exactly **32 real neural model
calls**, pretrained external pinned 2B versus frozen H136 64-step LoRA,
**0 optimizer steps**. Owner Windows checked source/adapter SHAs and raw
native response-to-point consistency, then executed **16 actual
model-proposed mouse down/move/up drags** on authentic unchanged app builds,
with success scored exclusively from persisted application state.

| Frozen H138 out-of-generator full apps | External pretrained 2B | Frozen H136 LoRA |
| --- | ---: | ---: |
| START within real dragged card/handle | 6/8 | **0/8** |
| END within destination column bbox | 0/8 | 0/8 |
| Actual persisted task success | **0/8** | **0/8** |
| React dnd-kit English | 0/2 | 0/2 |
| React dnd-kit Arabic | 0/2 | 0/2 |
| Native HTML5 board English | 0/2 | 0/2 |
| Native HTML5 board Arabic | 0/2 | 0/2 |

Exact paired full-task p=**1.0**, no successes in either arm.
This is a **failure of external full-app transfer for this 8-task sample**.
There is no defensible claim that the frozen 64-step synthetic-trained
research LoRA generalizes broadly: despite H136 15/24 full tasks on same-
generator heldout and H137 17/36 on library wrappers, H138 is 0/8 on
two independent complete applications. START also regressed in this
sample from 6/8 external pretrained to 0/8 LoRA (tiny n; exploratory).
Flagship Hala 35B model has NEVER been changed by H136–H138.

**Decision: NO-GO for promotion.** H138 screenshots and results are now
exposed for diagnostic analysis only; do not train on them, select a
new checkpoint with them, tune exact prompts on them, or call them a
release regression suite for improvement claims. Future H139 needs a
substantively NEW, app-diverse legally usable TRAIN corpus with multiple
real independent GUI implementation frameworks and newly preregistered
private out-of-distribution test applications. Compare frozen base and
adapter with true START acquisition, END finding, and full persisted
tasks. Include bilingual tasks, long-horizon recovery and H127
independent human review before claiming useful Hala skill.
No paid compute, release, publication, deployment or merge.
