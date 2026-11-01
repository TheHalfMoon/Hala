# H339 — Source custody and rights-admission review packet

**Status: Git source-provenance evidence only; COPYRIGHT/SCREENSHOT TRAINING RIGHTS NOT APPROVED.** This review packet contains **no screenshot, user data, action coordinate, rendered image, prompt/gold, model prediction, model weights, or private Kaggle receipt**. Google Elephant–Goldfish Model (EGM) upstream design standard is followed for evidence, explicit alternatives and unclosed decisions, but **no independent isolated Goldfish reviewer has completed a review**.

## 1. Original source families now requiring TRAIN approval

| Cohort | Original HTML / Git blob | Git introduction / immediate parent | Source browser qualification | Candidate capacity | Ownership/ML screenshot rights |
| --- | --- | --- | --- | --- | --- |
| RailKit H325 | `h325/railkit.html` / `743b5e1b54774a1deea4dbe6bfbe0613adead7e4` | `3303396b77dc7671cd97f425c644f71b6b0b664e` / `9c636b7c73679d8422db13a15e3fe7a298b39606` | H325 original source browser Chromium/Edge | 16 scenes; 32 native action-stage messages; 8EN+8AR | **UNREVIEWED — DO NOT TRAIN** |
| SignalDesk H338 | `h338/signaldesk.html` / `3d787aa3aafb999d342cbd3685c20d0f34d5fcd8` | `ae183d4531ccf1ada8c98098416e38e715332cd6` / `5aca7ceb8e4e5d86d736dd83683a60ef84d2d3f2` | H338 exact-head 34/34 GitHub checks PASS; 128 scenes / 256 unique decoded RGB source frames on both Ubuntu Chromium and Windows Edge; separate Windows owner 128-scene full test PASS | 128 scenes; 256 action-stage messages; 64EN+64AR | **UNREVIEWED — DO NOT TRAIN** |

Combined **144 scenarios, 288 candidate action messages, 72EN+72AR**, highly correlated within each fictional GUI family. These are NOT **288 independently diverse real-world tasks** and do not confer any source-family-independent OOD competence.

Git history alone proves neither legal ownership nor training rights. Both original source commits are observed **not cryptographically signed**, the project LICENSE Git blob is pinned to the original Apache-2.0 text; a generic repository license does not independently prove copyright title or a rendered-media machine-learning training grant. The local Git custody check requires each introduction commit to be an actual ancestor of the current exact HEAD, its immediate parent to lack that source, and the LICENSE/source Git blobs unchanged from introduction through HEAD. H329 and H339 HTML parsers separately reject external assets, CSS URLs, JS network loaders, source mutation. Failure refuses the entire combined packet. No downstream source document or prompt can self-authorize TRAIN.

## 2. Distinct approval decisions REQUIRED before H199 admission

An authorized rights reviewer (not the model, not the code author merely asserting a Git identity) must record documentary evidence for **each source family** on these exact independent questions:

1. Copyright/title/derivative origin: all relevant contributors identified, any donors/derivatives/external snippets checked, and their actual grants/permissions attached or linked. Git commit author strings alone cannot answer this.
2. Screenshot/image provenance: confirm generated browser raster imagery belongs to an authorized image-training scope; discuss fonts/system rendering and any residual rights, and whether redistribution of the images or LoRA artifacts is allowed.
3. Permitted use: explicitly approve **noncommercial local/Kaggle FREE GPU SFT on those rendered screenshots and native action labels**, or deny. Distinguish commercial/release/public weights scope from local training. Conditional/unclear scope is **DENY/DEFER**, not implicit consent.
4. Privacy and purpose: no patient, user account, credential, sealed blind OOD, third-party quarantined dataset or private-model action escapes custody. H293/H294 cannot be bypassed.
5. Training/DEV/OOD family split: RailKit and SignalDesk may be **TRAIN candidates only**. Public/exposed BeaconBoard is **not a fresh blind test**. New never-seen disjoint DEV and a separately sealed blind OOD family require independent sources, consent and preregistration.
6. Immutable candidate release manifest: exact Git head, both source Git blobs, approved owner/reviewer decision IDs, permitted use scope, generated frame validation count, source/visual-family separation and a SHA-256 of a canonical rights-approved release manifest. The approval record must be authored by the reviewer and signed/approved externally; **H339 deliberately does not generate a fake signature, rights decision or release manifest.**

Only after those decisions are independently **APPROVED** should a *new, reviewed Draft* change `h199/native_action_sft.py` `ADMITTED_TRAIN_MANIFEST_SHAS`, and even then all real FREE-T4 training and model development require a separately frozen processor/adapter/SFT protocol, full-source mask checks, privacy-safe aggregate reporting, and head-to-head adapted vs genuine pinned pretrained Qwen with independent heldout OOD. Nothing in H339 approves or starts those steps.

## 3. Concrete, reviewable independent status packet

`h339/original_source_rights_evidence.py` must return:

- Two verified source introduction lineages + Apache-2.0 Git blob unchanged and zero embedded external assets/JS network loaders, without treating those facts as ownership proof.
- Capacity counters (2 families, 144 scenes, 288 candidate messages, 72EN+72AR), with explicit review approvals = **0**, official H199 admitted manifests = **0**, model generations = **0**, optimizer steps = **0** and training use/release = **FALSE**.
- Fail-closed errors on source drift, insufficient Git history, false source ancestry, false rights approval, official TRAIN/DEV/OOD promotion, extra model generations, or false claims about crypto signatures.

Run locally against a clean **full Git history** checkout with `HALA_H339_GIT_QUALIFY=1` and:
```sh
python -m unittest discover -s tests -p test_h339_original_source_rights_evidence.py -v
```
Actual first-party full-history checks are enforced in `.github/workflows/h339-original-source-rights-evidence.yml` on Ubuntu and Windows; GitHub CI produces **no screenshots, artifacts or training datasets**.

## 4. EGM decision record and remaining blockers

- **Decision now:** prepare verifiable Git custody and explicit rights-review checklist, do **not** admit/train/release any cohort.
- **Alternative rejected:** equating Apache-2.0 LICENSE/unsigned Git author with independent image-ML training approval (leaves legal gap).
- **Alternative rejected:** admitting data by setting H199 to a locally generated hash, without reviewer documents (unsafe governance bypass).
- **Alternative deferred:** training on 288 candidate samples immediately; real learned improvement cannot be measured without approval plus truly disjoint DEV/OOD, and synthetic scale is small in model training terms.
- **Milestone accepted for this Draft:** actual two-source Git custody on both OSes and all source-only/adversarial tests PASS, immutable source counts and explicit nonadmission. **Milestone not accepted:** legal rights, training, independent fresh Goldfish, generalizable model competence, release.
