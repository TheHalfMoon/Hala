# H23 combined delta review

Reviewed repair handoff and affected source/tests against the initial findings. No source edits or GPU/model work.

## Must-fix

- **P2 — `hala/serving/lifecycle.py:144-150`, `evaluation/serving_lifecycle.py:290,299`: reap evidence can be false after cleanup timeout.** `close` clears the only stored child handle before either wait succeeds. If terminate/wait and kill/wait both time out (a relevant failed CUDA-worker scenario), it raises with `engine.process is None`; the fault record subsequently calls that `child_reaped=true`, and the final close cannot retry the lost handle. This can falsely satisfy the newly documented successful-reap prerequisite for overlap qualification. Retain the owned handle until exit/reap is actually observed, record explicit confirmed-reap state/error, and prevent restart while an owned child remains unreaped. Keep cleanup bounded and close streams/control in an appropriate finally path. Add a CPU regression for double wait timeout and later retry.

## Resolved / clean

- Original active-CUDA instrumentation finding: source repaired. TorchDispatchMode places a fresh event after real active-model CUDA matrix operations. The private control query selects an event recorded before the timestamped transport Event.set and observes it still pending later. Because an event cannot become incomplete again, that ordering provides a prospective proof route for work pending at cancellation receipt. Completed/missing events remain nonqualifying. No actual remote overlap has been demonstrated; all qualification flags stay false.
- Partial failure evidence: repaired with incremental progress callbacks before/after tasks, probes and reloads, with fault trace/ack captured before cleanup and cleanup protected against progress persistence failure.
- Font/Pillow/RAQM checks precede initial allocation and every restart in parent and child. The exact source commit and full clean Git gate are enforced before spawning/loading.
- Updated freeze `20b57ebb7e66b59f480cb89cb7fd363e671d212d48249ec99356856bc16730ae` verifies against the 53-file source superset, including package initializers and heldout fixture dependency.
- No new parsimony or scope issue found.

Validation: 16 focused CPU tests passed in 5.445 s; exact updated freeze verified. Additional CPU mock reproduced the cleanup issue: both waits raise TimeoutExpired, kill called once, `child_reaped_claim=True`, retained handle false. No GPU/model downloads or actual CUDA qualification.

panel: full delta contribution — correctness ✓ fresh-context Codex combined judge · parsimony ✓ combined judge · product ✓ combined judge · security separate/root-owned · Jev/Graft root-owned
