# H18 independent data-integrity / evidence review

Verdict: must-fix evidence trust gaps. The preparation-only execution boundary is intact: exact disabled config, no new runner/model/network/job path, seven hard stops, training_ready/training_executed/quality_qualified always false. No model authority or H14/H15 production contract change found.

Scope: seven new H18 files in Hala-adaptation; supporting existing source loader, SFT loader/converter and H14 protocol inspected. Fresh independent session-model PStack integrity/security judge; d1 not used as a security gate. No repository edits, commits, GPU/model runs or network work. Report alone written outside repository.

## Must-fix

- **P1 — training/adaptation.py:68-89,157: candidate media identity is self-authenticated.** Replacing an admitted image with an arbitrary valid red PNG and updating its media hash, row image hash and JSONL hash is accepted; output still says every_media_hash_and_rights_validated=true and original_synthetic/CC0. Prompt/action/source reconstruction does not bind image bytes to the reviewed original candidate. Pin the reviewed original manifest/JSONL/media identities against a trusted source-controlled inventory, or independently reconstruct and verify the qualified rendering. Add a coherent image+digest replacement rejection test. This audit's narrow existing-artifact scope makes a pinned inventory sufficient; no generic importer is needed.
- **P1 — training/adaptation.py:90-119: recorded H14/H15 evidence lacks a trusted identity boundary.** A probe replaced all 64 H14 raw outputs with private-target centers and recomputed every stored score; audit accepted and reported 64/64 rather than measured 30/64. Rescoring proves consistency, not authenticity of measured outputs. The H14 task digest also trusts its adjacent mutable freeze manifest. H15 compares report bytes only to a mutable independent record. Pin the previously reviewed H14 task/freeze/result and H15 report/independent-record byte identities, retaining score recomputation as a second check. Reject coherent paired replacements, not only stale hashes or changed scores. This prevents audit output laundering fabricated measurements while preserving all existing frozen bytes.
- **P2 — training/adaptation.py:152-153,165: unvalidated metadata becomes audit claims.** Changing manifest label_scope to `Fully validated bound Mind plans`, raqm_qualified=true and excluded_records=0 is accepted and echoed despite the actual 58 English first-action rows and unavailable RAQM. Changing independent H15 exact to the string `FULL QUALITY CERTIFIED` is also accepted and echoed as independent_comparison_exact. Validate exact expected metadata/types for this historical candidate and limited H15 evidence, or derive the output labels from checks. The trusted evidence pins above also reject altered historical claims; keep explicit Boolean/type and semantic checks to avoid misleading future maintenance. Add these negative cases.

## Worth considering

- tools/audit_adaptation_data.py leaves a previous output on a failed audit; docs explicitly warn users to remove stale output. An atomic-write/failure marker or source/run identity would make automation less likely to treat old evidence as fresh. This is not a launch bypass and is documented, so not a merge blocker here.

## Verification

- `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m unittest tests.test_adaptation -v`: **7 passed**, 8.774 seconds. Parent-provided full 145-test result was not repeated.
- Four isolated TemporaryDirectory negative probes: coherent image/digest replacement **accepted**; false candidate metadata **accepted**; false independent H15 exact string **accepted**; fabricated H14 outputs plus recomputed scores **accepted, 64/64**. Each retained training_ready=false. No repository fixture changed.
- Existing negative tests reject unpaired row/media tampering, train/split/language relabeling, missing rights, H14 changed scores/exclusion, unsafe config overrides. Those clean results do not cover coherent evidence replacement.

## Reviewed SHA256 identities

```text
badcc4997a5a78616bf9cf013856fd1a78823a2a991ecfd997c13b90c06fddbb  training/adaptation.py
62470ba78d81c44243ea89598d52f77d90770ac206a9caab495af766ba2b2ab7  tools/audit_adaptation_data.py
b47aa540e0e3cc976918da637f2bb5a6c40e9f99154d13a44401d1ce1ca9c328  experiments/adaptation/shared-multitask.json
cb7070dd2ae42b136476b13c0c010a57d89c4e41109f6ec0e5198ff9cf112f35  experiments/adaptation/readiness-audit.json
a6a783a57938f127f612c39d33a72a6a88a3ac62219d58b655e804a9b99ffca8  tests/test_adaptation.py
e6f49e5e2e877ba07b5de11e961060cacd4b4bd241645937890190a6c0740cb2  docs/H18_ADAPTATION.md
891dd32fd6e16ad4369f8f6b4a56a7c3abfa13c511056fa5d0e9620641062aab  specs/H18-adaptation-readiness.md
```

panel: full — integrity/security (fresh session-model Codex judge); correctness/parsimony/product and genuine Jev/Graft remain parent-owned gates. No d1 approval or learned-quality claim.
