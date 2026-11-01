# H334 — Full 12-case paired preloaded visual prompt extension (NOT RUN)

**Source-only integration and invented tests. No H334 GPU/model inference has
occurred.** This adds a single prospective, callable comparison that can
exercise the real H323 preloaded screenshot-conditioned Qwen model policy in
both H331 arms, but only when a separately authorized owner-private caller
supplies an already-loaded pinned model and valid source/browser.

## Research question

Does H331's opt-in first-click screenshot text-matching instruction reduce
wrong-button/focus failures on the **same** original publicly exposed
BeaconBoard tasks relative to the unchanged control? H324's one Arabic
failure (0/1) is only a motivating observation, not evidence of an
Arabic-wide defect. H326 ruled out simple x-axis reflection of correct
reference coordinates; it did not establish model visual localization.

H334 extends the stacked H332/H333 four-case matched study (the H333 private FREE-T4 payload is NOT submitted) to all 12 exposed original scenes, and **inherits H332's source-owned decoded RGB first-frame reproducibility check before each paired arm**. This is not a new blind source family.

## Frozen paired design

- Exact first-party HTML Git blob:
  `6e8467e123c7f0fbac77ea553622ab8e659f9c16`.
- All 12 **publicly exposed** original H319 scenes, six English and six
  Arabic RTL; fixed scene order `0,6,1,7,2,8,3,9,4,10,5,11`.
- Each scene is run twice in its own freshly isolated offline browser
  context, with `control` and `visual_target_rtl` using the same
  source/task and preloaded model. Three cases per language run control
  first, three candidate first; no outcome-driven selection or stopping.
- Only the first-click model-visible instruction differs. Stage-two
  prompt, grammar, normalized coordinate mapping, source scoring, and
  native actions are unchanged.
- Strict, bounded, aggregate-only outcomes: paired success discordances,
  failure/abstention counts, model-time generation and native dispatch
  counts, stratified by language. No per-scene record or raw prediction
  leaves the callable; an invalid or failed arm aborts without partial
  output. No files or network access are performed by the H334 wrapper.

### Actual invocation (owner-private, **not authorized by this PR**)

`run_paired_exposed_preloaded_probe(browser, source, processor, model)`
in `h334/frozen_full_paired_preloaded_probe.py` calls the real
`h323.preloaded_dynamic_bridge.run_preloaded_dynamic_source_research`
24 times. The function does **not** load/attest model weights, verify
free GPU quota, pin dependencies, or authenticate a private kernel receipt.
Those remain external mandatory preflight requirements. The caller must
also freeze model revision
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`,
decoding seed/configuration, original source SHA, noncommercial owner-private
FREE T4 eligibility, private output handling and unique trial identity.
Do not submit a job merely because this module exists.

This is **not** a blind/source-disjoint OOD study: all 12 tasks are
already public/exposed, and H324 has previously scored two of them.
A prompt-only difference cannot demonstrate LoRA learning. With 12 paired
tasks and shared source family, uncertainty is large and cases are
correlated; report paired discordance counts without generalization claims
or post-hoc cherry-picking. A real scientific gain still requires
independently lawful admitted TRAIN/DEV, new independently custodied
source-disjoint never-exposed blind OOD, genuine paired BASE/ADAPTED
task-state evidence with family-aware inference and safety/abstention.

## Offline invented-only qualification

```sh
python -m unittest discover -s tests -p test_h334_full_paired_preloaded_probe.py -v
python -m unittest discover -s tests -p test_h331_optin_visual_target_prompt.py -v
```

The eight H334 tests patch H332's real browser source audit only for invented-only CPU qualification; actual model runs retain the source audit. They use fake safe summaries to verify paired counterbalance,
aggregate arithmetic, rejection of raw/forged/extra fields, and no partial
receipt on a failed arm. **They do not execute Qwen or establish skill.**

Preserve H199 admitted TRAIN as empty, H173 scored OOD sealed, H212 barred,
H250/H251 rights quarantined, H293/H294 private safety blocks untouched.
No public private media, model output, coordinates, task gold, receipts,
weights, merge, deployment, release, or paid services. Keep stacked DRAFT.
