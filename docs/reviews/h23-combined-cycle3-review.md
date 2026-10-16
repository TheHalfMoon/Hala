# H23 combined cycle-three delta review

Scope: sole retained-worker stale-output repair. No source edits or GPU/model execution.

- **Must-fix: none found in this delta.** Explicit `usable` is initially false, becomes true only after a fresh exact ready handshake, and is cleared before fault handling and every close attempt. `_exchange` rejects an unusable/absent owner before serializing/writing its pipe or consuming queued output. Retaining an unreaped handle therefore cannot reuse old results. Successful reap leaves it unusable until a new start constructs a fresh queue and completes its handshake.
- The rejection releases the engine lock without entering cleanup or overwriting retained ownership. Prior confirmed-reap state and bounded retry behavior remain intact.
- No new correctness, parsimony or product issue found. Actual CUDA overlap, resource exhaustion and runtime recovery remain unqualified; source/CPU evidence supplies no actual-runtime qualification.

Validation: `rtk python -m unittest tests.test_serving_lifecycle -q` — 18 tests passed in 5.650 s, including retained-owner stale queue rejection with no pipe write/queue consume, successful reap still rejecting inference, and fresh-start recovery. Freeze `4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a` verified.

panel: full cycle-three delta contribution — correctness ✓ fresh-context Codex combined judge · parsimony ✓ combined judge · product ✓ combined judge · security separate/root-owned · Jev/Graft root-owned
