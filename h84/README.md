# H84 — Evaluations must match the exact model and evidence

**Stacked on H83 Draft PR #67**, parent SHA `8d8a83aa2f25528f2778b7ad25c0ca80f38e1ac7`.

## Measured orchestration defect

The H61 `step_evaluate` and `step_official_eval` paths reused a completed evaluation **solely because** `run-result.json` or `official-eval-run.json` existed at a fixed output path. Following resumed training, especially after H82 recovered an optimizer checkpoint, an old adapter evaluation could be silently presented as evidence for a new checkpoint. Changing official domains or `limit` could also reuse stale scores.

## Source-first H84 change

Keep the original H60 public DEV and H61 official evaluation GPU scorers unchanged except for additional official receipt identity/evidence fields; no new model runner is invented.

H84 namespaces DEV evaluations by the **full current verified immutable checkpoint SHA256** and official evaluations by a deterministic full SHA256 of model revision, frozen-base/adapter choice, entire adapter checkpoint digest, requested ordered domains, official limit and generation token cap. Prior evaluations are preserved under their historical paths; different optimizer steps or settings create different output directories without overwriting receipts.

**On cache hit**, H84 verifies:
- DEV: H60 adapter receipt schema, complete 24-case run, correct public DEV source SHA, fixed model ID/revision, original prompt/coordinate protocol, approved provider marker and real-inference status, exact current checkpoint SHA and adapter weights digest, and unmodified raw-attempts file SHA.
- Official: actual H61 receipt schema, fixed model ID/revision, requested domains/limit/token cap, exact adapter weights SHA and adapter-loaded variant; exact domain-set summaries plus fresh SHA256 verification of every stored official raw-attempt and score file.
- An unqualified/stale/corrupted or incomplete cache fails closed **before weight load** and is never overwritten. New results are independently validated after the original runner completes, before the H61 orchestration records their metrics.

The official runner now writes `requested_domains`, `requested_limit`, `requested_max_new_tokens`, plus SHA256 maps for each domain's raw attempts and score JSON. Old receipts without these fields will not silently pass the stronger cache gate; original artifacts remain untouched. Synthetic tests exercise cache hit, new checkpoint, changed official arguments, forged report digest/model identity, tampered raw/score files, legacy cache quarantine-by-namespace, missing/partial evidence and zero GPU runner when rejecting unsafe cached state.

No training, GPU allocation, private provider upload or measured GUI quality is claimed from these tests. Only real zero-cost T4 execution can establish actual model performance.
