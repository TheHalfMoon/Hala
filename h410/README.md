# H410 — no trust-on-first-use, nonce-bound authenticated source reviewer

## Why: H407's local review is forgeable

H407's separately injected `VisualTargetReview` checks a screenshot, exact user task, canonical model action, unique target count and bounding box before the unchanged H201/H203 native action executor. In REAL first-party owner Windows Edge, one source-known button was clicked and duplicate source buttons were refused, and the private H407 PR #437 exact head achieved **22/22 hosted Actions success**. But the proof is a local **unauthenticated Python object**. A bad/model-controlled reviewer callback could simply declare one target; this is NOT a trustworthy production authorization boundary. Actual H409 private Qwen3-VL-2B 128-step synthetic dual-family LoRA improved one-target clicks 7/16→14/16, yet **worsened false clicks on duplicate targets 5/16→6/16**, so a reliable independent source-review approval is not optional for real autonomous use.

## New optional signed proof verifier (not a replacement for original test APIs)

`h410.signed_source_review.run_authenticated_reviewed_policy` wraps the existing H407 screenshot-bound test-only review path with a **separate in-memory Ed25519 public-key verifier**. Every model-proposed click requires an independently issued `SignedSourceReview` that matches:

- A pre-enrolled **trusted reviewer key ID** (SHA256 of the exact 32-byte raw Ed25519 public key) + registered identity; neither the candidate envelope nor the model may introduce a new trust key.
- An unguessable, fresh 128-bit (32 lowercase hex characters) nonce created by the host for EACH candidate action. The nonce is made available to the separately isolated signing callback, but **never** to the prediction callback.
- Exact sha256 of the model-visible screenshot PNG bytes, exact UTF-8 user task, canonical parsed action, and exact registered source URL.
- Explicit `visually_verified_single_target` role, exactly **one** matching visual target, a finite in-bounds visible rectangle **containing** the proposed click. No drag/keyboard/scroll through H407.
- Strict integer issue+expiry timestamps in Unix seconds, valid at checking time with lifetime no greater than 30 seconds.
- Genuine Ed25519 signature over fixed-key-sorted canonical JSON payload, with signature bytes/base64 checked and cryptographically verified using only the **pre-trusted** public key.
- The ORIGINAL H407 screenshot/page guards then recapture the page and perform H201/H203 pixel-equality/URL/viewport checks before any native mouse event; no cached frame or public media persistence.

A model abstention sends ZERO browser events. A wrong, stale, replayed, expired or forged signature or unknown signer **always aborts**, with sanitized error text hiding reviewer internals.

### Inert / fail-closed production state

**`TRUSTED_SOURCE_REVIEWERS = MappingProxyType({})` is EMPTY by design.** There is NO onboarded independent authenticated visual reviewer, no signing private key embedded/generated in production, and **all actual production calls fail before screenshot, model inference, reviewer call and browser input**. Source code is intentionally **inert until a separate, genuinely qualified authorized signer is enrolled via governance**. Test-only Ed25519 keys are ephemeral within the unittest memory and never committed. The existing H407 unsigned source REVIEW API is test-only and unchanged, not an approved bypass for protected production actions.

This mechanism authenticates a *reviewer's cryptographic assertion*, **not** whether the human correctly observed the screen, legal rights to process a screenshot, independence of the reviewer, security of the signer's computer, source-webpage authority, or a race-free action transaction. A malicious or compromised previously enrolled signer could sign an incorrect claim. The callback still receives the PNG and must be separately isolated, consented and network-confined. No claim of a deployable clinical or reliable fully autonomous computer agent is permitted.

## Test-first owner Windows evidence

New tests failed RED on missing H410 module, then after implementation **9/9 full local source+crypto tests PASS**. Tests cover empty production trust registry before capture, synthetic memory-only Ed25519 sign/verify for a source-approved click, unknown signer, wrong nonce/replay, wrong PNG, task, action or URL, incorrect target count or offscreen box, expired approval, malformed signature, redacted callback exception, stale screenshot, and abstain with zero events.

`python -B -S -m unittest discover -s tests -p 'test_h410_signed_source_review.py' -q`: **9/9 OK with eight expected crypto-dependent skips** and the critical always-on default-empty-trust hard-stop test PASS. The full suite requires optional `cryptography` for ephemeral signature tests; no GPU, cloud, real-world screenshots or weights needed. H407/H203 prior tests must remain green on this stacked source branch.

No trusted reviewer public key, signing key or signed real-world approval was added. No patient data, external benchmark or real third-party GUI included. Keep this PRIVATE DRAFT, never merge protected main without explicit separate legal/security release approval.

## Hala real release NO-GO remains

H199 independent lawful original rendered GUI TRAIN manifest SHA allowlist is EMPTY; issue #384 independent rendered app screenshots/title/derivative model weights legal review remains OPEN, issue #385 genuine never-exposed independent bilingual EN/AR source-family-disjoint OOD custodian remains OPEN, no independently trusted action reviewer onboarded, and H409 actual model generated unsafe duplicate-target clicks in **6/16** first-party invented audit cases. Neither cryptographic proof structure nor temporary local test keys can override these blockers.
