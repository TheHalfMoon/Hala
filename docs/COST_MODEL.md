# Executable cost scenarios

Run `python -m experiments.cost.calculator --table` for all 25 founder-requested user-count/price scenarios. Supply `--inputs scenario.json` to replace the deliberately illustrative defaults. None of these prices is adopted.

Active GPU-seconds, additional retry calls, cold worker loading, warm idle residency, storage, transfer, verified-success denominator and failure overhead are explicit. GPU-seconds already sum worker time: concurrency does not divide their price. Offered concurrency load can flag impossible average capacity; burst queueing and tail latency remain unknown without arrival measurements.

Architectures A–F can each supply their own input file after a paired evaluation measures calls, memory, latency, retries and verified success. Do not compare architectures by invented model performance. The calculator refuses to label a scenario as measured by supplying a run name; genuine experiment manifests need a separate evidence review. No GPU price or inference cost has been qualified.
