# H338 — SignalDesk: 128 original bilingual two-step TRAIN candidates

**STATUS: SOURCE-ONLY TRAIN CANDIDATES; NOT ADMITTED FOR TRAIN, DEV OR OOD.**
**EGM:** see [Google EGM-based scaled-data feature specification](../docs/designs/hala-scale-firstparty-data-egm.md). Prepared according to upstream feature specification and implementation-readiness standards, but NO isolated fresh-context Goldfish review was actually run; `REVIEW_BLOCKED`, not approved.

H338 authors a **new, entirely fictional offline 960×720 SignalDesk GUI** with horizontal dispatch rows and inline CSS/JavaScript; no third-party media, private patient information, real user credentials, APIs, remote fonts, network calls or model weights. Its source HTML exact Git blob `3d787aa3aafb999d342cbd3685c20d0f34d5fcd8` is verified before every source browser use. Distinct original GUI family from prior exposed BeaconBoard (H319) and earlier synthetic two-by-two RailKit (H325).

## Scale and diversity, without inflated claims

| Original source cohort | English scenes | Arabic RTL scenes | Total scenes | Two-stage source action messages |
| --- | ---: | ---: | ---: | ---: |
| H325 RailKit (existing) | 8 | 8 | 16 | 32 |
| H338 SignalDesk (new) | 64 | 64 | **128** | **256** |
| Combined **candidate** capacity only | 72 | 72 | 144 | **288** |

H338 is eight times the original H325 count, and its source differs visually, but the 128 tasks are highly correlated because they use the same family. They do **not** represent 128 independent product use cases or independent OOD examples.

The new frozen Cartesian product has 2 languages × 4 button orders × 2 color themes × 2 row densities × 4 target signals = exactly 128 source scenes. A purely generic visible scenario number makes all first screenshots distinct without revealing target labels. After an actual browser click on the intended button, the input appears and gets focus; a second screenshot is captured; then a distinct invented `SIM-NNN` code is typed natively. Every task and label is from the source only. Exact input/focus/four-record state is verified after each action.

Before a single trusted batch callback is invoked, **all 128 scenarios and all 256 action stages must pass**. The 256 images must have distinct **decoded RGB** SHA-256 digests; any source drift, wrong state, invalid action, duplicate image, browser capture failure, or wrong role aborts with **zero candidate delivery**. The one optional in-memory callback receives all 256 `(PNG, source_task_prompt, native JSON action)` triples and must return literal True. The caller must separately guarantee atomic rollback if it has any side effects; this module never writes sample media.

## Validation and limits

```sh
python -m unittest discover -s tests -p test_h338_original_large_train_candidates.py -v
```

Seven CPU contracts cover source bytes, scene product and EN/AR balance, denylisted roles, exactly-one batch handoff, final-scene refusal with zero output, duplicate RGB refusal, and non-Boolean callback rejection. A separate opt-in **actual original source browser** test runs every two-step source action under Chromium (Ubuntu) or Edge (Windows) when `HALA_H338_BROWSER_QUALIFY=1`, and requires **128 successful browser scenes and 256 distinct model-visible RGB stages**. The full test never opens Qwen, performs optimizer steps, uploads media or writes PNGs as files or CI artifacts.

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains **empty**. First-party Git authorship of HTML does not prove independent copyright/derivative screenshot rights. A reviewer must separately verify custody, source family provenance, rights to rendered-image training, role separation and immutable manifest; only then a separately reviewed owner-approved H199 admission can happen. The model is NOT trained by this PR, no adapter gain is measured, and neither H338 nor H325 is a fresh blind OOD evaluation set. True Goldfish review and owner release decisions remain blocked; H173 sealed OOD, H212 barred source, H250/H251 rights quarantined assets and H293/H294 private retrieval remain unchanged.

No paid GPU or inference, no Mac, no main merge, no release/deploy, no model/media publication or private output download.
