# H402 — author-synthetic missing-target/abstention stress test

H401's owner-private 2B native point output achieved 18/18 original developer-authored colored-grid click hits after 16-step H399 LoRA, versus 3/18 unadapted. **That result is too narrow to qualify an autonomous computer-use agent.** H402 attempts to falsify overbroad claims using a newly constructed first-party-only set of 24 wholly invented 224×224 rectangle images: 12 have a red/green/blue requested target with nonstandard locations, backgrounds and paraphrased instruction prompts; 12 intentionally have **no requested target** and demand exact JSON abstention.

Each of the positive/absent strata contains 6 English and 6 Arabic prompts. Baseline and H399 trained private adapter are assessed on the **same** inputs using actual deterministic multimodal generation; metrics are exact target box hits, strict JSON/invalid outputs, correct abstentions, and **false clicks on absent targets**, reported by EN/AR and by target-present/absent status. The private adapter SHA and 16-step receipt SHA must match previously verified values; no labels from a genuinely hidden independent evaluator and no original third-party GUI screenshots are available.

## Verified result — Kaggle private T4 COMPLETE

Real source-only owner Kaggle V1 paired generated-click evaluation (base upstream Apache-2.0 Qwen3-VL-2B vs exact previously trained H399 private LoRA) returned:

| Strict generated output metric | Base Qwen 2B | 16-step private LoRA |
|---|---:|---:|
| Positive invented targets: correct box click | **0/12** | **12/12** |
| Absent invented targets: correct explicit abstention | **0/12** | **10/12** |
| Total correct action or abstention | **0/24** | **22/24** |
| Invalid strict JSON | **24/24** | **1/24** |
| False point clicks on absent targets | **0/12** | **1/12** |

For the adapted model: **EN 12/12** total correct, **AR 10/12** total correct; all 12 positive target clicks are correct across languages. Of 12 absent-target prompts, adapted model has **10 explicit abstentions**, **one WRONG real click**, and **one malformed output**; the two failures occur in Arabic conditions. Base model returned nonconforming strict JSON on all 24 new paraphrased instructions: its zero reflects exact protocol failures, not proof it could never perceive the target.

**Science/safety interpretation:** The adapted 2B shows significant source-only *invented grid* native output transfer to paraphrased coordinates, but fails an abstention safety requirement even in toy data; DO NOT deploy unreviewed native actions or claim generalization to third-party applications. For any future real computer-use agent, enforce independent action-validity checks, no-target ambiguity refusals, and permission/reversibility policy outside the model. **Explicit NO-GO for action autonomy, independent OOD and flagship Hala release.**

Exact owner Kaggle output evidence `h402/verified_receipt/h402-private-synthetic-negative-eval/receipt.json` is SHA256-verified by `h402/validate_receipt.py`; original SHA256 `420ad5c0b868c425e6dfb55d0c1dff107d8c61ff8a69866b5d668ad221aa868a`. Local source + receipt safety **9/9 PASS**, and stdlib-only version **9/9 OK with one PIL-only skip**. No model weights in source repo.

The test is **author-exposed, small, toy synthetic, not third-party sealed OOD**. A high score would not qualify original real-world Hala; the two failures are preserved explicitly rather than suppressed.

Run URL: https://www.kaggle.com/code/abdulazizshehri/hala-h402-negative-native-generalization

Private metadata `is_private:true`; Kaggle source only prior private owner H399 LoRA output; `dataset_sources: []` and `model_sources: []`; Internet enabled only for pinned public Apache-2.0 Qwen3-VL model and Python dependencies. No paid GPU, no public model weights, no personal/patient data. Source-only tests use stdlib and optional first-party image generator; a SHA-verified receipt is mandatory before reporting outcome.

H199 lawful GUI TRAIN allowlist remains empty, independent rights and derivative ML licensing review issue #384 open, independent never-exposed bilingual OOD custodian issue #385 open, and Hala release remains NO-GO.
