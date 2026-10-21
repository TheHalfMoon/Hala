# H49 — Original Arabic public-services + English FHIR visual grounding

**Date: 2026-10-21. Research-only, noncommercial, original synthetic offline GUI dataset. NOT a real Najiz/Absher/clinical user journey, learned model result or production authorization.**

Source branch: stacked on H48 first-party Chromium SFT [PR #36](https://github.com/TheHalfMoon/Hala/pull/36), current parent signed `54baccb55681ea12e5d08eed4809d095c2fee4fd`. Preserves all historical H14/H23/H41/H46/H47/H48 original source, frozen manifest, negative model scores and dedicated queued H47 Kaggle notebook. Uses only authorized Windows device Abdulaziz, no Mac, paid API or local model weights.

## Domain selection: separate language policy and real-world caveats

| Synthetic original family | Language | Split | Inspiration / observable skill |
|---|---|---|---|
| `justice_execution` | Arabic RTL only | TRAIN | Public Najiz service categories: request lookup, filing, evidence, scheduling, notifications; original paraphrased fictional labels |
| `traffic_services` | Arabic RTL only | TRAIN | Public Absher traffic and vehicle service categories: insurance, impound, appointment, driver license, citations, registration |
| `fhir_resources` | English only | TRAIN | HL7 FHIR **R4** resource selection: `Observation`, `DiagnosticReport`, `Patient`, `Encounter`, `MedicationRequest`, `Practitioner` |
| `fhir_bundles` | English only | TRAIN | HL7 FHIR **R4** `Bundle.type`: `searchset`, `transaction`, `document`, `message`, `collection`, `history` |
| `municipal_portal` | Arabic RTL only | PUBLIC DEV | Original fictional municipal permit application, distinct app/template/family |
| `research_lab` | English only | PUBLIC DEV | Original fictional non-patient research-lab workflow, distinct app/template/family |

**Official source references — vocabulary and semantics only (no scraped site HTML, logo, screenshot or proprietary asset):**

- Najiz official public services: https://najiz.sa/applications/landing/
- Absher official public portal and service guide: https://www.absher.sa/
- HL7 FHIR R4 Observation: https://hl7.org/fhir/R4/observation.html
- HL7 FHIR R4 Bundle: https://hl7.org/fhir/R4/bundle.html

All training labels, page markup, tasks, paraphrases, panel styles and screenshots are newly authored first-party CC0-1.0 fictional mock content. The presence of real product/service names in these **research notes** is descriptive only. Screens deliberately say they are independent research simulations, not official services. No government ID, plate, user record, medical patient, clinical data, credential, portal login, OTP, payment, case submission, CAPTCHA or session automation is collected or accessed. No real public website was opened by the capture script: only local `file:` pages were allowed, with every other request aborted. There is no authority to operate on real accounts.

## Delivered experiment assets and independent verifiers

`h49/factory.py` deterministically authors **96 offline HTML cases** across six fictional app families, 16 examples per family: 64 TRAIN (32 Arabic, 32 English) and 32 PUBLIC DEV (16 Arabic, 16 English). Each task presents six plausible service/resource options, shuffled across target positions, and an independently paraphrased semantic goal. No privileged target index/DOM box is included in a model-facing prompt.

`tools/h49_offline_capture.js` captures original 1024×640 Edge Chromium PNGs using pinned local `puppeteer-core 25.12.0`, verifies all six controls are fully visible within the unobscured viewport, then performs **one real browser DOM click on the intended option** and on a different distractor after resetting the page. These are **local mock-browser DOM clicks**, not actual real application or native OS actions. The resulting `h49/assets/capture.json` records every raw image digest/size, six measured DOM rectangles, correct selected key and wrong-key rejection. It does not make network requests to Najiz, Absher, HL7 or other domains.

`h49/audit.py` independently reconstructs the original source, checks original HTML **byte-for-byte**, PNG signatures/dimensions/SHA256, browser control text, labels and geometry, 96 golden and 96 wrong-option postconditions, full target click uniqueness, language/domain family isolation and explicit no-real-service claims. It exports create-only:
- `h49/assets/model_inputs.json`: 96 model-facing public synthetic input descriptions **without** target index/key/DOM truth
- `h49/assets/evaluator_only.json`: 96 labelled evaluation oracle cases, stored separately; **public and not blind**
- `h49/assets/authoring_sft.jsonl`: 64 train-only window-pixel click completions
- `h49/assets/sft_manifest.json`: source/capture/JSONL/media digests, rights, language/family and nonqualification metadata
- `h49/assets/integrity.json`: counts of the actual browser replays and source checks

Training input ingestion **reuses** the existing `training/scripts/train_adapter.py` with an explicit H49 schema and an H49-only byte-for-byte provenance-bound source loader. The H48 route remains supported and the H14-frozen `training/artifacts.py` is entirely unchanged. No unqualified model weights or tokenizers are downloaded.

Repro from repository root (only in a clean new output path, never overwrite original evidence):

```powershell
python -m h49.factory --root PATH_TO_EMPTY_NEW_OUTPUT_ROOT
$env:PUPPETEER_CORE_PATH = 'PATH_TO_INSTALLED_PINNED_PUPPETEER_CORE_25_12_0'
node tools/h49_offline_capture.js PATH_TO_NEW_OUTPUT_ROOT
python -m h49.audit --root PATH_TO_NEW_OUTPUT_ROOT
python -m unittest tests.test_h49_domain_grounding -v
python -m unittest discover -s tests -q
```

The original pinned output `h49/assets` is create-only. Keep original evidence unchanged, compare new independent runs by digest and report source/runtime/browser version changes. Never silently replace receipts or upgrade pinned captures.

### Genuine negative during build

The very first Edge capture attempt **failed** on `h49-traffic_services-01`, because the sixth button was partially offscreen and the attempted click did not register a DOM postcondition. The failure and independent root-cause replay are kept in `reviews/h49-original-capture-first-failure.md`. Repair compressed the original sidebar row heights, added explicit **fully onscreen DOM geometry admission**, and regenerated this new H49 source only before successful capture. The original H48/H47/H14 assets were neither touched nor refrozen.

### Scientific limits / next steps

These are original simulated single-click tasks, not a license to operate Najiz/Absher accounts and not a real OSWorld 2.1/ScreenSpot-Pro score. TRAIN and public DEV families are separated, but shared authorship, public oracle Git files, similar component grammar, publicly known goals and screenshot layouts mean **no independently blind heldout or generalization claim**. Arabic human review of transliteration, legal service terminology, and RTL accessibility remains outstanding.

**No actual H49 model training has occurred.** Before LoRA/adapted-model comparisons: verify the intended baseline SHA/tokenizer/processor and provider license, free compute terms and quotas, pre-register a *new independently sequestered* family/app-disjoint bilingual TEST set and score identical checkpoints under the same hardware/harness. The existing remote trainer still blocks Kaggle for eligibility and requires an exact real failed baseline; this work does not relax either condition. The original H46 0/8 outcome is retained honestly, H47 queued kernel must not be resubmitted, no commercial release or competitor-win claim is authorized.

CI, full serial local tests, genuine Graft graph integrity and optional independent D1/PStack/Alibaba semantic reviews must be stated from **actual** logs at the final signed SHA, never assumed from a draft or earlier head.
