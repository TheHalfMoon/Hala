# H55 — Real Original Browser Native-Drag Public DEV

**2026-10-21. A FIRST-PARTY 24-scene bilingual public DEV environment with actual Edge pointer replay. This is NOT independent blind TEST, model inference or training.**

## Actual source and browser work

24 independently authored original slider screenshots across six fictional scene families, completely distinct from real H Company DragOn TRAIN screenshot media: `ar_civic`, `ar_records`, `ar_logistics`, `en_fhir`, `en_design`, `en_inventory`. Exactly 12 Arabic and 12 English. Arabic scenes use original fictional Saudi-style public service topics but do **not** access or imitate Najiz/Absher source screens; FHIR is English. No real accounts, third-party screenshots, EHR/PHI, live services or official benchmark TEST labels.

`h55/public_drag_dev.js` pinned `puppeteer-core==25.12.0` and an **actually installed headless Microsoft Edge** at original 1024×640; outbound network requests are rejected. It created the public original source HTML, image PNG screenshots, and source/replay receipts in `h55/assets` without overwriting anything. Each screen has a real movable slider handle, a visible target mark, varied layout/title/geometry and English/Arabic instruction. A real pointer-down/move/up from the **correct handle** to **correct endpoint** advances the fictional state. Real track-background wrong origin and wrong endpoint drags must not complete it.

**Actual results: 24/24 real browser drag completions, 24/24 wrong-endpoint rejections and 24/24 wrong-origin rejections.** An independent Python source/PNG/geometry/replay digest auditor `h55/audit.py` checked exactly 24 unique 1024×640 genuine Edge PNGs, source HTML hashes, actual pixels and claimed state results, six disjoint fictional families and balanced 12 Arabic/12 English.

- Original first-party source SHA256: `707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6`.
- Actual raw Edge replay receipt SHA256: `cbff4d7d4db896b4a378435ffe69118c5a3d1f11e982391b934d5ad6a798d85a`.
- Original scene source, raw PNGs, source authorship plans, user goals, golden coordinates, and genuine Edge receipts are all committed to PUBLIC GitHub because they are fully first-party and PUBLIC DEV, unlike the third-party original DragOn TRAIN source, which remains **PRIVATE Kaggle only**.
- Original H55 DEV screens are **never added to H50/H52/H53 TRAIN or to Kaggle TRAIN bundles**.

## Scoring future real Hala predictions

`h55/score.py` validates the source-bound exact 24 raw action outputs and scores native ordered drag start/end point correctness by language. This is **geometric diagnostics only**. `h55/replay_predictions.js` separately takes those **unchanged raw model outputs** and executes them with real Edge mouse pointerdown/move/up to measure what the actual fictional GUI accepted and rejected, not merely whether JSON parsed or approximate coordinates match. It does not invoke the model or manufacture predictions.

The required input is one JSON object:
```json
{
  "schema": "hala.h55.public-dev-raw-drag-attempts.v1",
  "source_sha256": "707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6",
  "origin": "provider_raw_model_outputs",
  "cases": [
    {
      "id": "h55-ar_civic-00",
      "image_sha256": "EXACT_ORIGINAL_SCENE_SHA256",
      "raw_output": "UNMODIFIED_JSON_STRING_RETURNED_BY_THE_MODEL"
    }
  ]
}
```
The input above is **schematic only** and must contain exactly all 24 original source IDs, without duplicates, with original frozen PNG SHA256, and no source-privileged golden answers. It cannot be admitted as real Hala outputs unless raw provider model inference receipts and frozen weights actually exist. Both evaluation tools explicitly report `blind_test=false`, `training_executed=false`, `independent_model_inference_provenance_verified=false` and `official_benchmark_comparison_qualified=false`.

### Real replay harness qualification — NOT neural evidence

For harness tests ONLY, 24 deterministic *synthetic oracle* predictions were temporarily generated from the author's own source DEV golden endpoints; **none are claimed to come from Hala or any neural model**. Actual Edge pointer replay accepted **24/24** on these positive unit inputs and **0/24** deliberately wrong-endpoint unit inputs. Raw output receipts are preserved with `origin=synthetic_unit_fixture`, `synthetic_fixture_only=true` in:

- `h55/evidence/oracle-replay.json` SHA256 `f98c977964613dc96d28dc0defa96d4613e3725a35b556029542ac36a066bf64`.
- `h55/evidence/wrong-replay.json` SHA256 `7e9abe4cca1d701eeaa1f0b84a2cf7248777daeb90e60a6dd104bc1f7f91b354`.

These real browser replays prove **tester mechanics** only. They say NOTHING about learned Hala correctness or model improvements. `tests/test_h55_public_drag_dev.py` validates all source screenshots, Arabic/English balancing, no test-source training, raw JSON schema, source-image pairing, start/end proximity metrics, forged action rejection and fixture-versus-model provenance.

## Reproduce on permitted Windows Edge machine

```powershell
$env:PUPPETEER_CORE_PATH = 'PATH_TO_PINNED_PUPPETEER_CORE_25.12.0'
node h55/public_drag_dev.js h55/new-source  # create-only new folder
python -m h55.audit --root h55/assets
python -m unittest tests.test_h55_public_drag_dev -q
# Later ONLY when actual raw untouched model outputs exist:
python -m h55.score --root h55/assets --raw-attempts /path/to/24-real-model-raw.json
node h55/replay_predictions.js h55/assets /path/to/24-real-model-raw.json /path/to/new-output.json
```

## Scientific gates NOT accomplished

The original H54 real Hcompany DragOn source `type=drag` LoRA loader has been verified for 87 JPEG/760 TRAIN labels but the model **HAS NOT** trained on it. No free authorized GPU session, no new adapter checkpoint, gradient steps, real base Qwen3-VL-2B ordered-drag predictions, bilingual neural success measurement, matched adapted-model comparison, official benchmark performance or competitor win. H46 original click target baseline remains **0/8** and is NOT a native drag baseline. H47 single original Kaggle notebook is still **QUEUED** at last check; never submit again or bypass H41's frozen font requirement.

H55 is deliberately PUBLIC DEV, so future official performance claims require a **separate independently sequestered BLIND TEST** on actually new applications as well as real model/provider inference and execution receipts. Do not describe 24 authored DEV mocks as 24 independent neural agent successes.

## Engineering CI/review status

Individual original first-party `h55/audit.py` validated all 24 real Edge screens; new focused Python test suite passes 7/7 on real H55 evidence and synthetic negative fixtures. The complete source suite, current-head GitHub CI, true Graft wiring graph, and optional independent D1/PStack/Alibaba semantic review require separately observed live results; do not claim them in advance.

## Actual Windows H55 source qualification (2026-10-21)

- **7/7 real-source H55-focused Python tests PASS**, with model-free synthetic positive/negative fixtures, original `reviews/h55-public-dev-focused-20261021.log` SHA256 `e59738dda3360231a7c539ffa59d293a1168f435b976afe1195012c777d23881`.
- Full Windows `unittest discover -s tests -q`: **659 total, 657 PASS, 2 existing SKIP, zero errors/failures**, actual 279.593 seconds. Original `reviews/h55-public-dev-full-suite-20261021.log` SHA256 `b9387687d4f5c324a9f22643b0177657ded102f36238adac037664436a493592`.
- Genuine Graft wiring-only build/check **PASS** with 253 files, 2,052 nodes, 5,476 edges. Deep Graft semantic tier NOT run. Original `reviews/h55-public-dev-graft-build-20261021.log` SHA256 `ef6a60618cf0d0bda90eacecd45a11377607a7c396c9f2609d9fcbce83449f1d`, check SHA256 `7a005a6e059c15de0ae59f3c0ad7650d616c8183c90c871e33fc6004c4a2f8ea`.
- Local installed Alibaba OCR preview available; D1/PStack usable local review commands not verified. Do not count these as independent semantic reviews unless actual results are performed.
- Exact new GitHub head CI still PENDING until new source is signed/pushed and hosted Python 3.11/3.12 push/PR checks complete.
