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
