# H18 independent integrity delta review

Verdict: prior three must-fix findings resolved for the deliberately narrow historical inventory. No remaining must-fix or new concern in this delta.

Reviewed `training/adaptation.py` SHA256 `9d1deee891828a79e5e85b7cdda232c8fe879655b5c40ec3f25c4528f56f3305`. Historical source/examples/manifest, frozen H14 tasks+freeze+actual raw result, and actual H15 report+independent verification now have separate trusted digest anchors. The pinned manifest binds all image digests; the existing loader verifies actual media bytes. Canonical image paths and strict historical label/Arabic/H15 semantics retain the original source/action checks. This is a fixed reviewed snapshot, not a generic rights importer or prospective training loader.

Independent rerun of all four original isolated negative probes against this exact source:

- Arbitrary valid red PNG plus coherently updated row/media/JSONL hashes: rejected, `Reviewed historical evidence changed: training/generated-candidate/examples.jsonl`.
- False Mind label scope plus RAQM qualification / excluded Arabic count claims: rejected, `Reviewed historical evidence changed: training/generated-candidate/manifest.json`.
- Independent H15 exact changed to `FULL QUALITY CERTIFIED`: rejected, `Reviewed historical evidence changed: experiments/runtime/h15-independent-checkpoint-verification.json`.
- All 64 H14 raw outputs fabricated from private targets with correctly recomputed scores (probe confirms 64 fabricated hits): rejected, `Reviewed historical evidence changed: experiments/heldout/h14-fp16-result.json`.

The untouched real snapshot still audits successfully: H14 30/64, training_ready=false and seven hard stops. Stored configured audit reproduces exactly as a JSON object, including config hash. No source edits, commits, models, network or jobs; temporary probe copies only. Parent's 13 focused / 151 full green results were not unnecessarily repeated; the independent delta execution targets the four previously demonstrated bypasses.

Additional checked identities:

```text
8b4156f70b80803f79aaff2705bfbe4dee8952bc5debdb68b9fa1789160f184b  tests/test_adaptation.py
672c03dd662bf803d2f381376aa24f1ae64ab53444e33a6f9fa47faae5895bfc  experiments/adaptation/readiness-audit.json
```

Prior worth-considering stale-output CLI behavior remains explicitly documented; not a blocker for this phase.

panel: full — integrity/security delta (fresh independent session-model Codex judge), original exploit probes rerun; Jev/Graft and other review bars remain parent-owned. No model-quality or launch authorization implied.
