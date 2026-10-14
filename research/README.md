# Hala research evidence

| Artifact | Purpose |
| --- | --- |
| `FOUNDATION_MODELS.md` / `../hala/models/registry.json` | Nine pinned candidates, exact licenses and architectures, weight-only estimates, processor/runtime risks and explicitly unknown Hala metrics. |
| `HOLO_DATA_AND_TRAINING.md` | Public Holo methods; four real bounded DragOn schema/image samples; benchmark trajectories quarantine and independent data method. |
| `LAVAGUE_ASSESSMENT.md` | Actual source review at exact commit, named components, typed-action/retrieval concepts, eval/provider/telemetry/completion hazards and maintenance limits. |
| `../docs/FREE_GPU_RESOURCES.md` | Official free compute restrictions and actual qualification blockers, including Kaggle commercial restriction and Colab's new supported CLI. |
| `evidence/` | Revision-pinned official metadata, file SHA-256 provenance, dynamic provider/release page snapshots and existence-only credential probe. No weights/full datasets/customer secrets. |

Sources are first-party: Hub publisher cards/configs/licenses and file trees, official source repositories and provider terms/docs. Vendor performance and training disclosures are marked vendor evidence, not reproduced performance. Provider public pages are mutable observations dated 2026-10-14. Model/file revisions are immutable SHA pins; their download hashes are recorded in each `provenance.json`.

Exa discovery used seven successful searches requesting 39 result slots across model releases, provider restrictions, Colab supported clients and other free compute. These are not 39 unique or independently verified sources; duplicated/lower-quality third-party results were excluded from conclusions. One earlier schema-only failed Exa call requested six slots and retrieved no evidence. Direct official metadata/API fetches supplied the model registry and dataset inspection. Research evidence is captured locally to make the resulting conclusions reviewable without trusting a search snippet.

No source reuse, provider signup, billing activation, grant application, Space deployment, local weight download, complete dataset download, local training or model-performance qualification was performed by this builder. Root separately owns runtime smoke experiments and external-action approvals; those results must not be inferred from this research.

To reproduce metadata capture, run `research/collect_metadata.py` and `research/inspect_metadata.py` with Python standard library; `research/sample_datasets.py` additionally needs Pillow for bounded public-eval image dimensions. Collection caps each response at 2 MB and allowlists small metadata files. Image samples are decoded in memory. `research/build_registry.py` compiles captured metadata; `research/validate_sources.py` runs offline pinned-file hash, no-local-weight and box-bounds integrity checks. These scripts do not install a model runtime or qualify performance.

Integrity run passed for nine candidate pins, 35 captured file hashes and four actual DragOn eval samples; report is `evidence/integrity-check.json`. Open blockers: allowed authenticated free remote runtime; exact dependency/model runtime qualification; external data asset rights; Holo4 budget/VRAM; Liquid legal-entity revenue applicability; independent English/Arabic/RTL/grounding/planning and verified-task cost measurements.
