# H22 TEST precision comparison

The remaining TEST profiles use a separately qualified free Kaggle T4 allocation, UUID `GPU-14ac1f01-0bec-e9da-d360-51d417657b16`. Completed DEV profiles remain retained on the earlier allocation; none was rerun. The new prospective hardware/order amendment was published before TEST inference at `9d055aeec6329909c91b91a5f144298365110819`, path `h22-test-restart/experiment.json`, SHA256 `21cab7460e729b4f02190d03ba368a468a330477e902d11f694e3a3e8aaf6c99`.

Scientific source remains `07591f109b6683c9302fcf890259ddf0bde91494`, freeze `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b`. The original shared adapter manifest remains `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d`; data, prompts, parsers, verifiers, package pins and budgets are unchanged. All120 original PNGs were restored byte-exactly. The same pinned remote base weights were verified before inference; no base weights were stored on the founder's PC.

Each profile runs in a fresh process, in fixed FP16/int8/NF4 order. Its raw evidence must be exported, published and publicly hash-verified before the next profile. TEST local indices0/1/2 correspond to the original planned TEST slots3/4/5. This explicit restart is separate from attempt1's unknown unretrieved outcomes and attempt2's completed DEV-only allocation.

| TEST measure | FP16 | int8 | NF4 |
|---|---:|---:|---:|
| Status |Completed|Unknown; session expired before export|Not started|
| Persisted complete tasks |2/8|Pending|Pending|
| English complete |2/4|Pending|Pending|
| Arabic complete |0/4|Pending|Pending|
| Semantically correct initial plans |8/8|Pending|Pending|
| Episode Mind correct |26/56|Pending|Pending|
| Action errors / calls |42/55|Pending|Pending|
| Recoveries |1|Pending|Pending|
| Unnecessary calls |0|Pending|Pending|
| Protocol-rejected visual proposals |0|Pending|Pending|
| Episode model errors |0|Pending|Pending|
| Repeated standalone probes correct |3/10|Pending|Pending|
| Global model calls |121|Pending|Pending|
| Wall seconds |231.873|Pending|Pending|
| Warm load seconds |5.011|Pending|Pending|
| Model/state storage bytes |4,258,275,904|Pending|Pending|
| Vision subset storage bytes |813,914,176|Pending|Pending|
| Peak allocated CUDA bytes |4,517,115,392|Pending|Pending|
| Peak reserved CUDA bytes |4,609,540,096|Pending|Pending|

FP16 saved-record audit passed9,048 checks with zero errors/warnings. All55 actions and frame occurrences were independently replayed.54 occurrences matched frozen PNG bytes; one reached-state image had no original frozen PNG and was not independently rerendered. Both English successes and the authored transient-failure recovery passed the independent persisted-state checks.

| Retained raw evidence | Immutable artifact commit | SHA256 |
|---|---|---|
| profile-0-fp16-test-evidence.zip (308219 bytes) |`9568d58c74f3e1326b55b04d304f643e2bf1b99f`|`e098a7631708aa50609e56ba7701c01a1c5cc5804b87f9994b6dded181abf471`|

Paths are under `h22-test-restart/`. FP16 result SHA256 is `4f00aea88962adcee085c5ab476e16d72bc8f9db12b897e40ed615f3bbbf6f51`. Independent audit, review and public round-trip records are retained in the same artifact branch.

The ten standalone probes repeat across splits and were already exposed in DEV; they are diagnostic evidence, not ten new untouched holdout cases. Authored workflow templates are small and dependent. Adaptive trajectories reach different states and prompts, so aggregate Mind denominators and run times do not measure identical-question accuracy or causal throughput. Fixed temporal order and warm disk/OS caches further constrain cost interpretation. New TEST and old DEV physical devices differ: no cross-allocation paired-cost inference is permitted.

Recorded model storage, allocator peaks, sampled whole-device readings, CPU RSS and cache file-content bytes are distinct quantities. Notebook/launcher processes and the allocated idle second GPU are outside model-process cost. Sampled maxima are not continuous hardware peaks; pointer/inode deduplication and hardware capture are not independently attested. Power telemetry is uncalibrated; no energy, efficiency or monetary-cost claim follows. TEST results are not used to tune or retrain the fixed candidate.

The interactive allocation expired during a subsequent multi-hour conversation pause. At2026-10-17 01:36:58UTC, the new VM had no experiment root, int8 status or raw archive and a different physical GPU. Int8 completion or failure is unknown; NF4 was never launched. The empty replacement allocation was stopped. The FP16 public archive remains intact. A separately reviewed prospective batch plan is being prepared; any full triplet replication for matched hardware must remain a distinct experiment. See ADR0009.

No precision/model adoption, native integration or production release is qualified. Founder retains merge and release decisions.
