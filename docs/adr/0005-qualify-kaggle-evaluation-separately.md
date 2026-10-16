# 0005. Qualify Kaggle evaluation separately

Status: accepted (2026-10-16)

## Context
The fixed 116-step shared pilot completed and its original adapter bytes were retained at signed archive commit 274024e14699599a51fca2c948af40819532e18c. Fresh CPU-session public retrieval and finite payload checks passed. Colab now refuses free GPU allocation. The user authorized an existing free Kaggle GPU account; actual Kaggle GPU/internet eligibility remains unresolved in the refreshed UI. At the start of this phase, evaluation and training guards permitted only Colab.

## Decision
Add a narrow, explicitly opted-in Kaggle evaluation guard with real platform markers, while retaining the original Colab-only training guard. Keep frozen data, prompts, generation settings, precision, package pins and candidate bytes fixed. Actual Kaggle environment/GPU/font/dependency qualification must pass before inference. Compare the retained candidate with an unchanged base under the same qualified Kaggle conditions; preserve the Colab baseline as the original training-selection record.

## Consequences
+ A separately qualified eligible free provider can complete the pending experiment without paid resources or lost assets.
- Supporting a second environment adds guard and provenance tests. CPU fakes, marker checks and user sign-in alone do not establish usable GPU eligibility or comparable costs. Provider changes cannot silently authorize retraining or imply quality improvement.
