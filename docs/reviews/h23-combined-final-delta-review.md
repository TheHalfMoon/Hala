# H23 final combined delta review

Scope: sole second-cycle confirmed-reap repair. No source edits or GPU/model work.

- **Must-fix: none remaining in this delta.** The exact owned handle survives both bounded wait timeouts; explicit `reap.confirmed=false` and TimeoutExpired evidence replace the earlier cleared-handle inference. Restart rejects the retained handle without spawning or implicitly retrying cleanup. A later close retries the real wait, requires an integer return code, records PID/return code and only then clears ownership and closes retained buffered streams.
- Fault progress and final result persistence now copy the explicit reap confirmation/error, including when close raises. No false successful-reap inference remains at the reviewed call sites.
- No new correctness, parsimony or product issue found. Previous source-level repairs remain accepted; actual CUDA overlap, allocator exhaustion and runtime recovery remain unqualified pending independent remote evidence.

Validation: `rtk python -m unittest tests.test_serving_lifecycle -q` — 17 tests passed in 4.944 s, including double timeout, retained handle, blocked restart and successful later retry. Updated freeze `063dde2eee1635ce1a52e40748ef76c06fb63a2fedb8a69da3322e269261015c` verified. No actual GPU qualification.

panel: full delta contribution — correctness ✓ fresh-context Codex combined judge · parsimony ✓ combined judge · product ✓ combined judge · security separate/root-owned · Jev/Graft root-owned
