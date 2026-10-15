# Frontier PR #1 independent security review

- Role: fresh-context PStack SECURITY judge, inherited session model.
- Immutable base: `00f6692383ca9e9cc147380b062676dd74932e83`.
- Immutable head: `9f4c26dbb4d4cefe15c4177e10f3e6976145af14`.
- Engineering diff: `work/frontier-engineering.diff`, SHA-256 `0b37ab7ccd72d4a7a0af7af98d2d83870f3f9ded292ea75fc4030aa7d9c96b63`.
- Inventory: `work/frontier-inventory.json`; all 365 snapshot file digests independently matched. Review and execution used `work/frontier-review-snapshot` and immutable `git show`, excluding later H14 changes.
- Disposition: **no must-fix security finding identified in the reviewed implementation**. This is bounded independent review evidence, not merge or production authorization.

## Findings as data

```json
{"must_fix": [], "worth_considering": []}
```

## Security bars and evidence

| Surface | Evidence and result |
| --- | --- |
| Authentication and tenant separation | `hala/serving/__init__.py:83` compares bearer bytes using `hmac.compare_digest`; `:245` authenticates every dispatched HTTP path; `:98` binds inference to the credential's configured tenant. Fixed public errors and `repr=False` credential configuration avoid observed token/request logging. Independent probes: all seven unauthenticated routes returned 401; suite verifies cross-tenant 403. |
| Model advice and execution authority | `hala/serving/__init__.py:110` reparses returned records against the request. `hala/contracts/proposal.py:132` binds request/tenant/task/device/session/generation; action validation enforces allowed vocabulary, method ceiling, observed targets, protection/redaction, frame/scale and target containment. `hala/contracts/responses.py` similarly binds plans and requires advisory-only routing. All 18 independent binding mutations rejected. Mock metadata cannot claim qualification; real backend remains unavailable. No native executor or authenticated real-model serving path exists. |
| Cancellation and bounded HTTP work | Exact fields, bounded strings/arrays and finite numeric parsing; duplicate JSON/header rejection; one request per connection with no transfer encoding or Expect. Absolute read/inference deadlines and cancellation are exercised. `hala/serving/__init__.py:345` admits before handler spawn; `:272` retains a shared lease while an inference worker runs. Tests verify timed-out/disconnected workers cannot prematurely free capacity. |
| Final SFT/media repairs | Directly inspected final `training/artifacts.py` and converter, rather than relying on the older H2-H11 report. `:55` rejects traversal, non-normalized paths and symlink components; `:110` checks train-only synthetic manifest/row/media identities. JSONL limits are 128 MiB / 10,000 records / 200,000 bytes per line; images are digest-bound, limited to 32 MiB and 16 million pixels, and verified as PNG before model allocation. `train_adapter.py:35` rechecks media before processor use. Original task export independently regenerates canonical sources (`data/factory.py:240`) and excludes holdout families. Tests reject changed media, paths, rights, schemas, duplicates and data/manifest drift. |
| Final checkpoints/deserialization | `training/checkpoints.py:79` validates metadata/state, exact artifact file set and byte hashes; `:97` validates the bounded normalized latest pointer and input identities. `:117` stages complete versions and atomically publishes one pointer, preserving the old pair on interruptions. `training/scripts/train_adapter.py:142` explicitly uses `torch.load(..., weights_only=True)`; `:189` writes adapter weights with safe serialization. Tests cover corruption, missing/extra files, identity changes, interrupted writer/pointer publication, and completed resume finalization without runtime/model import. |
| Remote model/helper paths | Linux and explicit experiment opt-in are checked before model work; provider markers are deployment diagnostics. Model revisions are immutable SHA values; both processor and model use `trust_remote_code=False` (`train_adapter.py:111`, `:117`; `baseline.py:120`, `:123`). Telemetry opt-outs precede Hugging Face imports; GPU probe sets them at module entry. Pinned candidate dependencies and all scripts retain unqualified status. Research downloads use fixed Hub sources, allowlisted metadata names and a 2 MB response cap; credential research records existence only. Helper subprocess arguments are fixed argument arrays, with no observed untrusted shell construction. Registry references to vendor `trust_remote_code=True` are research descriptions, not an executable loader. |
| CI/container and committed secrets | CI uses `pull_request` rather than `pull_request_target`, read-only contents permission, pinned action commits and a job timeout; no secrets or elevated publish step was observed. Container uses a digest-pinned image and non-root user. Both notebooks contain zero output cells. Pattern inspection of all 365 immutable files and the 4,788,515-byte base-to-head commit patch stream found zero candidates for OpenAI/Hugging Face/AWS/GitHub token patterns or private-key headers; values were never printed. Live CI and commit-signature/DCO verification are handled by the coordinating reviewer. |

## Verification performed by this judge

- Bundled Python, immutable snapshot: `python -m unittest discover -s tests -v` — **91 passed in 9.053 seconds**.
- Independent boundary probes rerun against that snapshot: **416 nested request substitutions**, 397 contract rejections, zero unexpected exceptions; **18/18 response binding mutations rejected**; **11/11 raw HTTP probes** returned expected status codes (unauthenticated routes, case-insensitive duplicate auth, folded header, unexpected GET body, Expect).
- Independently verified snapshot digests and the stated engineering diff hash before accepting test evidence.
- Direct review of final sensitive training code and its integrity/interruption tests; no model downloads, real training, paid compute, remote servers or external telemetry actions performed. Test HTTP listeners were loopback fixtures only.
- No source changes, commits or pushes made by this judge. Only this review document was added.

## Scope limits

Cancellation remains cooperative: a trusted future backend ignoring cancellation can retain its bounded slot. This review does not qualify that backend. Observation authenticity/freshness, live device authorization and final-state verification remain Aladdin/local-policy responsibilities; contract bindings are contextual checks, not authentication of observations.

Training configuration, manifests, output directories and checkpoint trees are operator-controlled offline experiment inputs. Digests detect drift; they do not independently prove source rights, measured baseline truth, signed provenance or malicious-writer exclusion. No generic untrusted checkpoint-upload or customer-data importer was found. Environment opt-in/provider markers are operational controls, not an adversarial isolation boundary. Real processor/PEFT/torch execution, checkpoint resume equivalence, resource fit and GPU efficacy were not executed or qualified. Third-party package vulnerability history was not exhaustively audited.

Secret inspection is pattern-based and bounded to the reviewed tracked snapshot and PR commit range; it cannot certify absence of every credential format or external/untracked secret. Global Graft telemetry configuration and live release gates are outside this judge's verification. No d1 result was used to satisfy this review; no Jev, managed Alibaba or Codex Security release approval is claimed.

panel: security ✓ independent fresh-context PStack judge; immutable PR #1 head; no must-fix security finding in bounded reviewed scope; live CI/signing and release authorization remain separate.
