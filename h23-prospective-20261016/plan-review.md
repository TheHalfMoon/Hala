# H23 prospective plan factual review

Plan checked against reviewed source/config at `18ad45deaa890c27c7250681e4bffe8b881e5690` (current relevant source/config has no delta from that commit); freeze `4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a` verifies.

- FP16 unchanged base, eager attention, exact model revision, no adapter, two bilingual three-step open/enable/save workflows and separate persisted-state verifier match source/config.
- Fault ordering matches: allocator-budget probe, disconnect, deadline, worker failure. Each destructive mode reloads the worker; the same two recovery workflows run once after all modes, not after every individual restart. The allocation is 8 MiB under a restored process cap, not a whole-device OOM test.
- The 1200-second outer timeout is a prospective **operator wrapper requirement**, not implemented by the H23 CLI. Source bounds include 300-second default load/reload, normal 60-second request deadlines, a 0.05-second deadline-fault configuration, and 250 ms private cancellation status capture. The external wrapper must also preserve/export artifacts on timeout.
- Apply the complete cancellation evidence rule from README during independent inspection: exact event-set receipt, real-model event recorded before receipt and still pending at later child query, **cancelled fault**, no contradictory generation-return evidence, and explicitly confirmed reap. The plan's abbreviated criterion should not omit those latter rejection conditions in actual qualification.
- Incremental task/fault/reap/reload result records are implemented. Any additional raw command logs require operator capture; child dependency/model stderr is deliberately discarded, so the plan should not imply full worker stderr capture.

No unsupported attention, fault-order, task or model-selection claim found. Execution remains prospective and conditional on all six H22 raw exports, safe owned remote cache preparation, and a fresh clean checkout of the exact reviewed commit. This review did not revalidate CI history or execute H23/GPU work; every actual qualification flag remains false.
