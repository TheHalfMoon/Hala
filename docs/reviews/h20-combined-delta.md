# H20 independent combined delta

PASS: previous P2 malformed-probability finding is closed. Range checks now precede `math.isfinite`. Independently parsed a valid JSON response under 5 KB containing integer probability `10**4000`; the repaired parser rejected it with `ValueError`, which the diagnostic already records as an error prediction. No uncaught `OverflowError` reproduced.

Replayed all 32 existing recorded request bodies/hashes and predictions, corpus/split/learned-model identities, all summary correctness/error/abstention/unsafe counts and median/P95 timings, 5,070-byte classifier size and four stdio wire hashes using the repaired parser. All match the original record. No new online calls, model execution or full-suite run occurred. The backup config hash and parsed comparison still confirm only the UTF-8 setting differs; no settings/credentials were printed.

Evidence identities remain distinct:

- Historical executed diagnostic source: `7260f83e484ce51d86a7278a93e29c16d16d24276a0c232ffe757279966bd7db`.
- Current reviewed parser repair: `2dcef6570c051db1e4fab455355796dfd83bde04e3538a4ff0f901047e257e49`.
- Unchanged historical comparison: `bc5fde251db206991c7675215f1a75cbd54829022171e8ae18a94527744e6d04`.

Independent offline probe: `work/h20_review_probes.py`, updated outside the repository to expect rejection and distinguish historical versus current source hashes; exit 0. No repository source or recorded comparison bytes changed by reviewer. Original 13/32 d1 correctness and seven unsafe suggestions remain measured historical results, not new qualification of the repaired source or learned routing adoption.
