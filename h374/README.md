# H374 — Make signed sealed-OOD custody date validation UTC-exact

**RESEARCH ONLY: no independent custodian enrolled, sealed test received, benchmark
admitted, training permitted, model improved or released.** Draft stacked on H373.

## Verified security correctness issue

The external signed metadata protocol H356 explicitly calls its two date
fields `signature_date_utc` and `expiry_date_utc`, but the production
verifier used `date.today()` (the local system time zone) for comparison.
H355's independent legal-review signed-date verifier already uses UTC-aware
`datetime.now(timezone.utc).date()`. H356 did not. Around midnight UTC,
a host in UTC+3 could reject a still-UTC-valid custodian signature early;
a host in UTC−7 could accept a signed receipt after its UTC expiry date.
This matters if an independently verified custodian is ever enrolled.

## Safe minimal fix

- H356 now reads the UTC calendar date from `utc_calendar_today()`, which
  is explicitly aware `datetime.now(timezone.utc).date()` on every host.
- Existing signature expiry bound (1..180 days), canonical date parsing,
  previously trusted reviewer/custodian signing keys, original evidence
  commitments, exact source-family counters, SHA and secret privacy
  restrictions remain unchanged.
- Existing H356 invented test fixtures now generate their labeled UTC dates
  with the same UTC-aware clock to prevent host-timezone flaky tests.
- New adversarial tests prove the clock explicitly requests
  `timezone.utc` across 23:59:59 -> 00:00:01 UTC and rejects altered,
  noncanonical or future-bound dates. When the free pinned Ed25519 test
  dependency is available, invented ephemeral in-memory signatures validate
  same-UTC-expiration-day acceptance, post-expiry rejection, future-date
  rejection, and that *even valid* signed metadata never changes TRAIN,
  blind OOD, model gain or release authorization.
- Dedicated GitHub 4-way Windows/Ubuntu Python3.11/3.12 workflow installs
  `cryptography==50.0.1` and **fails if the signature tests would skip**.
  It runs H374 plus inherited H356 and H355 signed-intake tests; it never
  generates externally trusted keys, hidden cases, credentials or weights.

## No false scientific advancement

The simulated positive test patches an invented ephemeral signing key into
the isolated test trust registry. The actual H355 legal reviewer and H356
independent OOD custodian trust registries remain immutable EMPTY, H199
approved TRAIN SHA manifest allowlist EMPTY; independent rights Issue #384 and
fresh unseen source-disjoint bilingual EN/AR OOD Issue #385 remain open.
A signed metadata claim is not independent legal or source-disjoint audit,
and it never qualifies blind scoring or learned Qwen gain. No LoRA training,
patient data, private imagery, GPU, model release or main branch merge.
