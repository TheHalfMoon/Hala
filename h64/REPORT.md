# H64 — H61 remote execution and Colab watcher durability

Status: **IMPLEMENTED; TARGETED TESTS PASS; REAL GPU NOT RUN.**

Stacked on H63 Draft PR #47 at `b05e1955500a272ca5a8126dcecda134287a18bb`.
No merge to H63/H62/H61/main and no provider or cloud operation.

## Two verified execution-breaking defects

1. **Incorrect Python subprocess result contract.** `h62.run_h61_diagnostic.run` returned `subprocess.CompletedProcess`, while all its callers destructured `(code, stdout, stderr)`. `CompletedProcess` is not a three-tuple, so even a successful provider session could not complete the operator-side path. The wrapper now returns exactly that triple.
2. **Incorrect Colab probe contract.** `h62.colab_t4_watcher.main --probe-only` unpacked only two values from `ProviderClient.existing_session`, which returns three. The probe now consumes the existing `name, hardware, error` fields and falls back to status only if the list has no hardware value.

## Two additional operational integrity protections

3. **Receipt success is verified, not assumed.** The operator runner removes stale downloaded receipts, requires both fresh remote files to exist and be nonempty, parses them, verifies the pinned repository commit and zero child exit, and admits the run only if the H61 run receipt reports completion, no failures, and the exact requested steps. A successful `colab exec` without successful downloads is **not** success.
4. **Restart preserves the original deadline and terminal state.** `ReceiptLog.restore` reconstructs the initial UTC timestamp and recorded stop outcome. A restarted process subtracts elapsed wall-clock time from the original 36-hour budget rather than getting a new deadline. Already-allocated or stopped watchers do not allocate or launch again. This does **not** create a distributed lock between independent watcher processes; the operator must still ensure only one instance exists.

## Measured local evidence

The original targeted test command, on a separately cloned Windows checkout:

```text
python -m unittest tests.test_h64_runtime_integrity tests.test_h62_colab_watcher tests.test_h63_visual_intervention tests.test_h62_collapse_probe -v
Ran 65 tests in 0.650s
OK
HALA_H64_TARGETED_EXIT=0
HALA_H64_TEMP_CLEANED
```

The targeted regression uses authentic Python `CompletedProcess` instances and simulated provider responses, including missing download, inconsistent pinned SHA, failed child, stopped watcher and expired prior deadline. It is NOT evidence that Colab granted a T4, that the machine trained weights, or that Hala performance improved.

## Honest remaining blockers and next step

- No authenticated Colab CLI was available on PATH on either presently connected Desktop Commander device at the last probe. This is an **environment observation**, not proof of provider unavailability or absence of a watcher on another machine.
- The free T4's current availability cannot be asserted from the CPU-only tests.
- No new H61 model checkpoint, private readback or matched live GUI accuracy is claimed.
- A remote operator should inspect live provider sessions/receipts *first*, then run a single pinned diagnostic on genuinely authorized free compute. Check that the watcher is actually alive and avoid duplicate launch.
- After the H61 model completes, run the independent H63 three-arm image intervention against genuine separate model inferences; never replace model reruns with static target permutations.

Do not publish or merge automatically until the branch's hosted CI succeeds and a reviewer inspects the change.
