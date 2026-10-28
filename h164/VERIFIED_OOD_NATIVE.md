# H164 — independently audited native task reward on TWO new original applications

**2026-10-28. Qualified one-shot exploratory OOD result; NOT a product release or broad generalization proof.**

## Actual source, model, and browser evidence

The source holdout was qualified *before* candidate neural inference and is now
**irrevocably consumed**. It contains 32 original Pell editor scenes and 32
original Pikaday calendar scenes, **32 Arabic + 32 English**, all 960 × 720,
with 64 independently exact PNG browser replays, original reference-native
success, and separate owner-private gold. These apps were excluded from H164
TRAIN and DEV. OOD manifest SHA-256:
`3991d8b5b200ef9e80c5567534a6637df3df3319686ae9fe2c05b85cd7e75fd4`.

Private Kaggle FREE Nvidia T4 inference generated 128 actual neural outputs
(64 identical scenarios × H161 old and H164 updated model), *zero* optimizer
steps. Source OOD ZIP SHA-256:
`6056b16ef9f6ef86ec2a514e7d71a6d2840d0930dcf770ffa198e1367f444a85`.
Old H161 LoRA SHA-256:
`72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c`.
Trained H164 LoRA SHA-256:
`43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
Trained new model ZIP SHA-256:
`4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da`.

The real owner Windows browser executed **128/128** frozen model-proposed
single pixel clicks inside the actual original app runtime, on identical
exact-replayed preaction PNG scenes. No gold targets, DOM selectors, source
oracle action coordinates or scripts entered the model/controller. Only the
original app's actual state after the model click was scored. The one-shot
owner-private consumption lock was created before any browser trial and
prevents reusing the holdout for retuning or repeated tests.

## Real native browser task outcomes

| Split | Previous H161 | Trained H164 | Count |
| --- | ---: | ---: | ---: |
| Pell editor, genuinely unseen app | 0 | **7** | 32 |
| Pikaday calendar, genuinely unseen app | 1 | **3** | 32 |
| English | 1 | **9** | 32 |
| Arabic | 0 | **1** | 32 |
| **Total** | **1 (1.56%)** | **10 (15.63%)** | **64** |

**Paired results: 10 new wins, 1 regression**, exact two-sided paired sign
`p=0.01171875` across all 64 cases. This is one exploratory, one-shot
cross-app test; do not read the p-value as proof of reliable generalization
across arbitrary applications. The per-app improvements are not equally strong,
and the language disparity is material. On 54/64 tasks the new model still
failed to complete the original application task; this cohort evaluates
**click actions only**, not unseen-app drag, typing, multistep navigation or
desktop workflow success.

The private reward receipt is
`private-h164/H164_TRUE_NEW_APP_MODEL_NATIVE_REWARD.json`, and the
one-shot lock is
`private-h164/H164_TRUE_NEW_APP_NATIVE_ONE_SHOT_CONSUMED.lock`.
Independent owner-side audit
`python -m h164.audit_ood_native --owner <OWNER_ROOT> --model-root <PRIVATE_GPU_OUTPUT>`
returned `H164_INDEPENDENT_TRUE_TWO_NEW_APP_NATIVE_REWARD_AUDITED` with exact
matching source SHA, adapter SHA, 128 GPU calls, 128 browser trials,
64 paired cases, per-app and language counts, paired sign p, lock, and
per-row boolean action/state outcomes. All private PNG, model weights,
raw predictions, gold labels and trial details remain **off GitHub**.

## Research decision

**H164 2B research language adapter has improved task reward versus H161 on
this frozen 64-case OOD experiment, but is NOT qualified as a generally reliable
computer-use model.** No flagship Hala 35B weight change or release.

- Never train/select/retest on Pell/Pikaday 64-case OOD, GridStack H163 or
  any historically consumed H140–H163 frozen evaluation source.
- Require a **fresh different family of original native apps**, registered
  before any inference, with novel layouts and distinct click/drag/type
  workflows and substantially stronger Arabic coverage; reuse frozen base
  training provenance, train-only new supervision and free compute.
- For spatial weakness, separately preflight conditional visual-projection
  adaptation on a fresh TRAIN-only corpus with a one-step T4 memory cap.
  Do not assert that such a change will be feasible or beneficial until it
  is genuinely tested. Compare new model and frozen baselines on unseen
  app-native task rewards, not just output format validity.

**No paid GPU, H140 sealed blind calls, public model/media/weights, main
merge or release. Keep PR #154 and PR #155 Draft.**
