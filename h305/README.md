# H305 — native browser dispatch accounting integrity

## Observed source-level evidence defect

H304 correctly prevented unsafe `type_text` dispatch, but its source
receipt still used `native_actions: 2` after one actual focus click
and **zero native type writes**. The legacy field had evolved into a
count of *policy turns*, not a count of browser actions. A first-stage
abstention similarly reported one "native action" despite dispatching
nothing. Treating that field as actual actuator count would overstate
execution and contaminate later model-evaluation receipts.

## Narrow, backward-compatible repair

- Keep `native_actions` for compatibility with older frozen research
  receipts, explicitly interpreting it as **model/policy turns**.
- Add `native_dispatched_actions` to the H296 source evaluator and
  H298 integration receipt, counting **only** actual browser click/type
  dispatches, not predictions, abstentions, or blocked writes.
- Independently fail closed unless the two counts agree with the
  focus/type dispatch flags, model-turn count, abstention and source
  write-refusal status. Reject booleans masquerading as integer counts.
- Preserve the H304 pre-dispatch write refusal and H301 terminal latch.
  Do not send the source-owned oracle, code, field ID, coordinates,
  or any private result back to the neural policy.

| Original source control path | Policy turns (legacy) | Native dispatches |
| --- | ---: | ---: |
| First-stage abstention | 1 | 0 |
| Wrong focus, source refused | 1 | 1 |
| Second-stage abstention | 2 | 1 |
| Wrong code, write refused | 2 | 1 |
| Correct focus and native type | 2 | 2 |

The counts do **not** attest a real pretrained model or a correct task.
All source and browser tests use fixed or stubbed model outputs, not
genuine new Qwen inference. Historical H300 is pinned to older code;
this additive field does not revise its authenticated exposed-source
12/12 result. H292 paired LoRA remains BASE 6/8 vs ADAPTED 6/8.

## Science and privacy boundary

No new official H199 TRAIN/DEV admission, independent blind OOD,
adapter gain, or release readiness. H173 scored OOD remains sealed,
H293 blocked output is not retried, and H294 blocked provenance write
is not bypassed. This is source-evidence integrity hardening only.
No protected-main changes, private media/weights, GPU, Mac, deployment,
or paid services.
