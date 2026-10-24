# H85 — Shared pinned Qwen cache across the first free-T4 training sequence

**Parent:** H84 Draft PR #68, SHA `ef46bb966912a36144815c2b525dc0a0e21e817c`.

## Concrete first-checkpoint bottleneck

Training used `/tmp/hala-model-cache`, H60 public DEV used `/tmp/hala-h60-model-cache`, and H61 official evaluation used `/tmp/hala-h61-model-cache`. Worse, H60 and official evaluation assigned `HF_HOME` **after** importing `transformers`, where Hugging Face cache defaults can already have been read. Consequently an approved free T4 could spend its limited lifetime fetching the same immutable Qwen3-VL base weights and processor repeatedly, and H74's intended warm-cache path might not match a subsequent evaluator's actual cache.

## H85 implementation

Source-first, no model loader rewrite. All eight existing **Qwen and processor** calls across the H61 adapter trainer, frozen baseline smoke, H60 public DEV and H61 official evaluation now explicitly pass the same resolved `cache_dir=/tmp/hala-model-cache/hub` to the original `AutoProcessor.from_pretrained` and `Qwen3VLForConditionalGeneration.from_pretrained`. A pure CPU-only `h85.model_cache.pinned_cache_kwargs` restricts calls to the unchanged original model ID and immutable source revision, and forces `trust_remote_code=False`. `HF_HOME` is also unified and set **before** heavy imports where feasible; explicit `cache_dir` remains authoritative even if Hugging Face was imported elsewhere beforehand.

The shared Linux `/tmp` cache stays within the current free Colab session: it is **not** a durable checkpoint, cross-session promise, Kaggle GPU run, predownload of weights or new storage service. Model revision, training weights, configuration, LoRA, dataset, optimization/evaluation scoring and user cost policy are unchanged.

## Evidence and remaining gate

Deterministic Windows CPU checks confirm the exact pin and cache path, fail-close wrong revisions/models, independence of a stale `HF_HOME`, and AST coverage of both processor **and** Qwen model loaders at all four production entrypoints, plus early environment setup. These tests do **not** measure a reduction in actual network traffic or verify GPU availability. H74's read-only pinned cache probe now points at exactly the same path actually passed to every original loader.

The next true hardware gate remains an allocated, eligible **free T4**, actual Qwen LoRA gradient update, optimizer-bearing immutable checkpoint with host/private SHA readback, and independent held-out Arabic/English computer-use evaluation. Keep Draft/NO-GO pending those facts.
