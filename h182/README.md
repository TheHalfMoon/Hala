# H182 — bind original OOD source reservation to observed Git history

This is **SOURCE-ONLY metadata hardening** on top of H181. It does not
capture or inspect any OOD GUI, admit TRAIN/DEV, train a model, or approve
an OOD evaluation. All historical OOD/DEV scenes remain permanently consumed.

H181's `source_reservation_sha256` is intentionally a hash of the fixed
public Gantt source identity. By itself it remains identical when the
owner's scanned historical Git checkouts change. H182 adds
`source_history_binding_sha256`, a second SHA-256 over the immutable
source identity, the H178/H179 in-memory historical Git identity digest,
and the aggregate observed checkout/repository counts.

Run the existing owner-only command **only on authorized Windows**:

    python -m h181.audit_owner_reservation --historical-root <authorized-history-root>

The output contains aggregate counts and two digests, never raw historical
repository names, local paths, media, prompts, gold, predictions, or weights.
For a reproducible comparison, save the digests in an owner-private
SHA-only audit record, not on GitHub. If a previously scanned repository
is omitted or replaced, the binding digest changes even though the fixed
public Gantt source fingerprint does not.

**This is not a completeness proof.** Deleted Git roots, copied or forked
sources, npm packages, old archives and prior image/RGBA SHA histories may
still be missing. A matching digest only attests to the *same observed
census*, not to every original source ever used. Never treat H181/H182 as
permission to open the reserved Gantt GUI, evaluate OOD, or use H173/older
consumed benchmarks for tuning. Freeze genuinely new TRAIN/DEV and a
one-shot OOD protocol only after independent full historical provenance
qualification. Keep PRs Draft; no merge, publication, release or paid GPU.
