# H146 — source vs heading point error: independent 4-way causal ablation

On **16 original MIT React tasks with never-before-used attempt IDs 102000+**
and four CSS-shifted layouts, the owner ran **64 genuine native browser
mouse drags**: four distinct conditions from independent fresh browser
contexts per original task. Every attempt's state was verified after a full
page reload, not merely matched against a coordinate rectangle.

Every condition used the same original visible source card and destination
heading. The proposed points always stayed **INSIDE their own visible
native bounding rectangles**, with deterministic ±20% source position
deviations and ±20% horizontal / ±32% vertical heading deviations.
No controller had access to the invisible native drop target DOM rectangle.

| Actual one-shot original browser actuator | Persisted task success |
|---|---:|
| Perfect source center + perfect visible heading center | **16/16** |
| Noisy source + perfect visible heading center | **16/16** |
| Perfect source center + noisy visible heading | **0/16** |
| Noisy source + noisy visible heading | **0/16** |

The 16 original tasks were balanced across Arabic/English × Doing/Done
(four cases per cell) and four shifted browser layouts (four each).
Three other cases were rejected because all four arms couldn't qualify,
plus eight cases excluded to maintain language/destination balance.

**Finding:** In this defined perturbation regime, **destination-heading
point precision alone determines whether the fixed direct-drag actuator
succeeds**; the same-size source-card point variation is tolerated.
H143 model-driven result **8/16 persistent tasks** on its own consumed
independent cohort remains unchanged. H145 staged 5/16 vs direct 0/16
on combined jitter is a different, already consumed cohort.

The next meaningful improvement is a **vision-observed, interactive
drop-activation controller**, or a geometry-tolerant approach, tested on
entirely NEW task IDs and frozen before seeing test results. Do not
calibrate/retest H143–H146 consumed episodes.

This H146 study used **no neural model calls**, **no SGD**, **no paid
GPU**, **no H140 sealed independent BLIND** and no main Hala 35B
change. Neither models nor datasets have been released. Exact owner
episode results stay in
`private-h146/native-four-arm-point-error-v1.json`.
