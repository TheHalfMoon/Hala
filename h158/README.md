# H158 — external original TodoMVC cross-application transfer gate

**State:** Fresh owner-private **16/16 original app reference tasks** passed
in the unchanged official MIT `tastejs/todomvc` ES6 JavaScript app,
pinned Git commit `ff43b02e59dfa604386bb382034b2cd07c2bcd8a`.
This is a DIFFERENT codebase and interface from MiniWoB++ and was
**not** part of H156/H157 training.

H157 achieved **20/72 (27.8%) → 60/72 (83.3%)** native task completion
against the same originally frozen MiniWoB++ DEV app, but did not yet
prove it transfers to other app implementations. H158 is **inference-only
cross-app evaluation**, no training from the new test cases.

## Genuine second-application frozen owner source

**Unmodified original** `examples/javascript-es6/dist/index.html` and
all its bundled JS/CSS, loaded locally in native Chromium without npm
install, internet providers, source edits, or a new hand-written app.
Source upstream LICENSE remains owner-private.

16 distinct 960×720 screenshots, all source-reference native clicks
succeeded based on actual app UI state:

| Task | EN | AR | Total |
|---|---:|---:|---:|
| Complete named todo checkbox | 4 | 4 | 8 |
| Click Active filter | 2 | 2 | 4 |
| Click Clear completed | 2 | 2 | 4 |
| **Total** | **8** | **8** | **16** |

Existing todo task rows were set up through **native UI typing and Enter**,
not injected directly into the DOM. Screenshot captured **before** action.
Original native point oracle is used ONLY to prove source QA, not as model
input or executable action in the actual test.

**Important original-app limitation:** This pinned ES6 app stores todos
in JS memory only; reloading wipes them. Score actual visible app state
after click, **NOT** persisted/reload state. No unverifiable persistence
claim should be made.

- Private source root: `private-h158/todomvc-source-once-v1`.
- Frozen original manifest SHA-256:
  `d242938ceacfe5ff48cedbcbc71b2efaf99b0b36813f410a289b1f8f591a9e44`.
- Source original MIT upstream Git SHA:
  `ff43b02e59dfa604386bb382034b2cd07c2bcd8a`.
- Owner source audit: `H158_SECOND_APP_SOURCE_AUDIT_PASS`.
- Private Kaggle evaluation dataset:
  `abdulazizshehri/hala-h158-private-second-app-click-v1`.
- Frozen starting H156 LoRA SHA:
  `732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6`.
- Frozen trained H157 LoRA SHA:
  `181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8`.

## Frozen paired neural gate

The private free T4 kernel should run **16×2=32 genuine GPU calls**
against the same source images using the H156 and H157 frozen external
research 2B LoRAs, without optimizer or prompt tuning. Actions must be
exact model-generated `click(x,y)` or `abstain`; any other output is a
failed/unsupported prediction. No source-reference point, DOM selector,
or original task state is passed into the neural controller. The owner
then tests resulting clicks in **two separately seeded original app
contexts** and verifies real UI state only after model action.

Metrics need EN/AR, task family, per-arm task success, paired gains,
regressions and uncertainty. Keep the source and all predictions/weights
owner-private; push only the code and aggregate outcomes. This holdout
may be small (16 cases), so do not overclaim real-world robustness
even on a positive result.

**No H140 sealed OOD BLIND use, no paid GPU, no flagship Hala 35B
weights changes, no merge, release or published model weights.**
