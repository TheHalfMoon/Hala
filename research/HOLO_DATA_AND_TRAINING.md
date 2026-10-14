# Public Holo methods and dataset boundaries

Research snapshot 2026-10-14. Official release pages are vendor evidence, not independent reproduction. Hala has no access to Hcompany's private corpus or complete training configuration.

## What public evidence actually supports

[Holo3 release](https://hcompany.ai/newsroom/holo3), March 31, 2026, describes synthetic navigation examples from human/generated instructions, programmatic out-of-domain augmentation, filtering and curated reinforcement learning. Its Synthetic Environment Factory uses coding agents to build web environments/tasks from specifications with end-to-end verification scripts. It reports a 486-task corporate benchmark, but does not publish the full private training set or a complete reproducible optimizer/hyperparameter recipe.

[Holo3.1 release](https://hcompany.ai/newsroom/holo3.1), June 1, 2026, focuses on mobile and cross-harness robustness, native function calling in addition to JSON, compact 0.8B/4B/9B models and quantized 35B artifacts. Its family benchmarks and DGX Spark/Apple results are vendor measurements in particular harnesses. They do not establish Arabic competence, Windows reliability or Hala serving cost. No full Holo3.1 private corpus/training recipe was found in inspected material.

[Holo4 release](https://hcompany.ai/newsroom/holo4), September 28, 2026, provides more concrete methods: 127B SFT tokens; successful task-factory trajectories approximately desktop 45%, web 14%, MCP/API 12%, mobile 3%; remaining data multimodal reasoning, grounding and text tool use/coding. It reports asynchronous online RL with `prime-rl` for two LoRA experts, desktop/web and terminal/MCP/API, merged with equal weights into the SFT model without more training. Rank, learning rates, optimizer state, reward implementation and exact training examples remain unknown.

The Holo4 Agentic Task Factory builds approximately 10,000 tasks (about 4k web, 3k MCP, 3k desktop/OS) from documentation, screenshots and real software. Its useful verifier gates are: untouched seed fails, golden state passes, near misses fail, and an agent solves through the actual interface. Failed attempts feed audit/hardening. Public benchmark trajectories are separately released for score transparency; they are not the SFT corpus.

Benchmark leakage caveat: Holo4 release says 480 of 600 public AutomationBench v1.0.6 tasks fall in the split from which training data were collected, with 120 held out. Its public-set score therefore must not be treated as wholly unseen-test performance. The same post reports harness engineering using OSWorld2 feedback and differing subsets/efforts across compared systems. Hala must freeze its own held-out tasks and avoid tuning against them.

## DragOn bounded inspection

Official dataset [`Hcompany/DragOn`](https://huggingface.co/datasets/Hcompany/DragOn/tree/3451c1fc3543c18fc3361a32bfe1f12cc69b64b5) pinned to `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`, declared Apache-2.0, English/French. Card reports 3,560,142 training tasks from 286,012 training images, 1,000 public evaluation tasks and a separate 1,000-example private test set absent from the repository. Training is four tar archives; public eval is per-image JPEG plus task JSON. Dataset Viewer `/splits` and `/size` returned HTTP 404 in this inspection; no full dataset was fetched.

Each task file is a JSON list. Observed fields: `intent`, `start_bbox`, `end_bbox`, `domain`, `subtype`, `ordered`, `image_id`, domain-specific `metadata`. Boxes use **original image pixels** `[x0,y0,x1,y1]`, not normalized coordinates. `ordered=false` means either endpoint direction yields the selection (text/sheet); `ordered=true` defines directional resizing/slider operations. Choose a legal point inside each target box, preserve direction where required and transform through the actual screenshot/processor geometry.

| Domain | Actual one-image sample width x height | Representation observed | Card's dominant resolution |
| --- | --- | --- | --- |
| text_highlight | 1650 x 1275 | French phrase selection; start/end boxes, full_bbox; ordered false | 1275 x 1650 portrait |
| sheet | 1920 x 1080 | cell_range with range_bbox and cell content metadata; ordered false | 1920 x 1080 |
| slide_resize | 960 x 720 | resize task; directional endpoint boxes | 1280 x 720, 960 x 720, 960 x 540 |
| slider | 1280 x 800 | directional slider task and metadata | 1280 x 800 |

Four actual public-eval task/image samples were fetched; ~1.02 MB JPEG bytes were decoded in memory for dimensions and not retained. Exact task/image URLs and sample JSON are in `research/evidence/Hcompany--DragOn/samples.json`. The text sample orientation differs from the card's dominant resolution; this does not invalidate the card's distribution claim, but makes a fixed-size importer wrong. Read actual per-image dimensions.

Repository-level Apache declaration is evidence of intended release rights, **not proof of every screenshot/font/document/underlying asset's rights**. Inspected card and sampled metadata do not supply complete source generators, asset attribution or acquisition licenses. Duplication, train/eval overlap, source-template overlap and overlap with OSWorld/ScreenSpot cannot be established from four eval samples. Repeated tasks per image are intentional; row-level random splits would leak visuals. Before accepting any train subset, audit source/asset rights, hash exact and perceptual images, and group splits by image/source/template/task family. Treat every public eval sample as eval-only; initial rights-cleared training data should be Hala-owned synthetic fixtures until external provenance is resolved.

## Holo trajectories are quarantined benchmark evidence

[`Hcompany/trajectories`](https://huggingface.co/datasets/Hcompany/trajectories/tree/ede2a362c99ddba0d03f439c47a0b8fea6abae22) pinned to `ede2a362c99ddba0d03f439c47a0b8fea6abae22`. Its official card says all **7,366** trajectories are runs behind Holo4 benchmark scores: OSWorld, OSWorld2, AndroidWorld, AutomationBench, PinchBench and Agents' Last Exam. Layout is `data/index.json`, `data/t/<id>.json`, `img/<id>/<step>.webp`; summary rows include model, benchmark, run, task, instruction, success, score, duration and steps. Steps contain reasoning/actions/results/screenshots. Only metadata/README were downloaded.

The repository declares Apache-2.0 but explicitly preserves upstream task licenses (OSWorld/AndroidWorld/ALE Apache, AutomationBench/PinchBench MIT) and masks identified credentials/PII. Masking does not prove all sensitive material is gone. Do not train on these tasks, screenshots, reasoning, derived labels or near-identical reconstructions and then evaluate on those benchmarks. Keep separate `evaluation-only` provenance and deny the source from training manifests. Use only permitted failure analysis/reproduction, never model superiority claims inferred from publication.

## Hala's independent, reproducible data method

Build owned synthetic browser/Windows-like controls with English/Arabic/RTL text, target identity, bounding boxes and deterministic state transitions. Generate selection, typing, drag, recovery, ambiguous requests, observation requests, rejection and abstention cases. Store fixture revision, seed, image hash/dimensions, template/task-family ID, label generator version, source rights and strict train/dev/held-out partitions. Validators must fail on untouched seeds and plausible near misses, pass on golden state, and inspect actual final state; an LLM saying success is not the label.

Use train-only templates/seeds to generate bounded successful trajectories. Keep visually related templates in the same split, store rejected/failed trajectories separately, and evaluate frozen held-out cases before LoRA experiments. A classifier under 600M or a deterministic router is an economical Reliance alternative. Reproduce verifier/task-factory methods from public descriptions, not proprietary corpora, public benchmark answers or unrestricted shell/MCP execution from a vendor harness.

Evidence capture scripts allow only small model/dataset metadata and bound each fetch at 2 MB. Initial trajectories API exceeded that metadata cap due to huge file inventory; the retry requested only sha/cardData/description and succeeded. No model weights or training archives were fetched locally.
