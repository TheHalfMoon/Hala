# H18 independent combined review

Verdict: must-fix before acceptance. Fresh-context correctness/parsimony/product review of the seven new H18 files against CLAUDE.md, PRODUCT.md and specs/H18-adaptation-readiness.md. No implementation edits, commits, model work or full-suite rerun.

## Findings

- **must-fix — correctness/product — training/adaptation.py:77–89:** Original row validation reconstructs the prompt and first action but omits screenshot identity. Swapping `image_path` and `image_sha256` between the existing same-family rows `intent-save-00-862846` and `intent-save-02-112562`, then recomputing only the JSONL digest in the manifest, is accepted by `audit()`. The loader validates media hashes and family membership, so both mismatched screenshot/label pairs pass. This contradicts the required rejection of altered rows despite updated JSONL hashes and weakens original-source data binding. Bind each row to its canonical original image path and immutable reviewed media identity; add a same-family screenshot swap regression. The implementation must also reject replaced image bytes with self-consistently updated manifest/row hashes if it claims original media identity, rather than merely integrity relative to a mutable manifest.
- **worth-considering:** none.
- **skip-it — parsimony:** No extra loader abstraction or configurable readiness system is needed; the small exact disabled config validator and fixed inventory audit fit this preparation-only phase.

## Confirmed clean

- Product scope remains one shared pinned base, one prospective rank-4 multitask adapter, disabled training/download/job flags, no datasets, no runner and seven permanent preparation hard stops. No separate-adapter implementation or learned-quality claim.
- Existing data is correctly described as 58 English first-action rows (20 semantic_invoke, 20 click, 18 set_value), zero Mind plans and zero normalized point labels. Existing dev/test fixtures are not presented as compatible adaptation evaluation data; Arabic remains excluded and shaping unqualified.
- Frozen original allocations and task text/action identities are checked through the existing source importer. H14 is held out, rescored at 30/64 (16/32 English, 14/32 Arabic), and its limited correlated-fixture scope is explicit. H15 is represented as two English engineering steps with independent exact comparison, expired runtime and no retained quality adapter.
- Future admission and baseline/dev/test selection requirements are concrete and keep H14 out of training and prompt tuning. No invented training counts or launch path.

## Verification

- `rtk proxy <bundled-python> -m unittest tests.test_adaptation -v`: **7 passed**, 8.627 seconds.
- Separate temporary-copy mutation reproducer: same-family image-pair swap + updated JSONL digest returned successfully with `first_action_source_binding_validated: true`. Temporary directory cleaned. No repository evidence changed.
- Full suite was not repeated; builder-reported 145-test result is documented but not independently rerun here.

## Reviewed source SHA-256

```text
training/adaptation.py badcc4997a5a78616bf9cf013856fd1a78823a2a991ecfd997c13b90c06fddbb
tools/audit_adaptation_data.py 62470ba78d81c44243ea89598d52f77d90770ac206a9caab495af766ba2b2ab7
tests/test_adaptation.py a6a783a57938f127f612c39d33a72a6a88a3ac62219d58b655e804a9b99ffca8
docs/H18_ADAPTATION.md e6f49e5e2e877ba07b5de11e961060cacd4b4bd241645937890190a6c0740cb2
specs/H18-adaptation-readiness.md 891dd32fd6e16ad4369f8f6b4a56a7c3abfa13c511056fa5d0e9620641062aab
experiments/adaptation/shared-multitask.json b47aa540e0e3cc976918da637f2bb5a6c40e9f99154d13a44401d1ce1ca9c328
experiments/adaptation/readiness-audit.json cb7070dd2ae42b136476b13c0c010a57d89c4e41109f6ec0e5198ff9cf112f35
```

panel: combined correctness/parsimony/product (fresh-context session model); separate security review is parent-owned and not satisfied by this report.
