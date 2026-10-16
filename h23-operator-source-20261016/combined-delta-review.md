# H23 operator combined repair delta review

Reviewed operator SHA `fff4235e6e81ac325b0456805d1d7d18c59a1d41a450482db7c82f1b37b3f1a9` and tests SHA `9902261a74f3f9923b4a4c02cbc2ae942235d0d1106c0594338102da030bad09`. Scope is the post-Popen settlement/retention repair only; no scientific source changes or actual operator/GPU execution.

- **Must-fix: none remaining in this delta.** Post-launch capture is protected by finally. Selector, read and log failures trigger bounded TERM/wait and KILL/wait handling of only the newly owned invocation group, close captured stdout, and record leader-reap plus group-quiescence separately.
- The normal export path now requires confirmed producer quiescence after any launched invocation. Failed settlement records `incomplete_cleanup` and emits a raw-retention diagnostic with no normal ZIP, avoiding the earlier live-producer evidence snapshot. Prelaunch failures can still export their static failure evidence.
- No new correctness, parsimony or product issue found in the repair. Reviewed helper SHA remains `d8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e`; the two-group/source/hardware barriers remain unchanged.

Validation: `rtk python -m unittest discover -s outputs -p test_h23_operator.py -v` — 12 CPU tests passed in 1.132 s, including selector/read/log error cleanup and failed settlement blocking normal archive. Linux process-group behavior and actual H23 execution remain unqualified; no GPU/model work, remote operation or publication performed in this review.

panel: operator delta combined contribution — correctness ✓ fresh-context Codex · parsimony ✓ combined · product ✓ combined · security separate/root-owned
