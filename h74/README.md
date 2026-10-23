# H74 — Immutable TRAIN source binding and offline model-cache preflight

Parent: H73 Draft PR #57, commit 5e52d2843e822a807fe3c9a33227949686a8e739.
Status: code and deterministic CPU tests, not an actual optimizer step or checkpoint.

## Confirmed source-identity bug

The actual H71 stage embeds source Git SHA acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe, but the old H62 launcher called the H67 restore validator with the *running-code* Git SHA. When the running code advances, the genuine historical archive can fail H67's SHA-bound admission.

H74 now requires three separate immutable values:
- --pinned-sha: reviewed executable Git commit.
- --stage-sha256: SHA256 of the original CPU stage archive.
- --stage-source-sha: the original historical data source Git commit.

The host supervisor -> local H62 operator -> remote pinned H62 source -> H67 restore chain propagates the two stage values; H67 verifies the stored index against the original data commit. The executable code is still independently checked against --pinned-sha. Missing or corrupt values fail closed BEFORE packages/model weights.

Example, **only** on a previously authorized free T4 with the H71 archive already uploaded and SHA-verified:

    python -m h65.safe_diagnostic --session EXISTING_FREE_T4 --pinned-sha REVIEWED_H74_40_HEX --destination PRIVATE_RECEIPTS --steps train --stage-sha256 4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a --stage-source-sha acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe --max-steps 1

Keep the exact H67 rederivation, Kaggle model-training restriction, strict checkpoint egress and evaluation gates. No code/weights/data publication.

## Source-first infrastructure REA: reducing first-checkpoint latency

H72's actual Colab T4 died during model preparation and left NO_PUBLISHED_CHECKPOINT. H61 sets HF_HOME=/tmp/hala-model-cache, an ephemeral path which is NOT populated merely because Kaggle CPU has verified its private source data.

The H74 read-only probe calls the official Hugging Face snapshot_download API with local_files_only=True and the exact pinned Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203. It checks indexed weight shard presence without creating a GPU session or downloading any weights:

    python -m h74.weight_cache_preflight --cache-dir /tmp/hala-model-cache/hub

CACHED_WEIGHT_FILES_CANDIDATE_ONLY is NOT model-load, processor, runtime, gradient or checkpoint qualification. BLOCKED_NO_VERIFIED_WARM_SNAPSHOT means no sufficient cached weights were proven at the target runtime. A warm cache on Windows or Kaggle does not prove the Colab GPU has a warm cache.

Relevant upstream reuse guides:
- https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- https://huggingface.co/docs/transformers/installation
- https://huggingface.co/docs/transformers/peft

## Scope, evidence, and next condition

H73 already has two independently verified private Kaggle CPU-source admissions, including offline admission. H74 tests are deterministic local fixtures, not proof of new GPU availability or training. At inspection time, colab was not found on the authorized Windows PATH, hence live T4 status is unknown.

Only after confirming compliant free compute and sufficient model weight availability: attempt ONE real optimizer update, independently verify and privately egress one checkpoint, then run held-out bilingual GUI evaluations. Do not claim an adapter exists until these gates pass. Keep stacked PR Draft.
