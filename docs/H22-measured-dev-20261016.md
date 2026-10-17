# H22 actual DEV precision comparison

All three prespecified DEV profiles completed with the same original retained shared adapter, frozen H19 inputs, runtime source `07591f109b6683c9302fcf890259ddf0bde91494`, exact package pins and physical T4 UUID `GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1`. Each raw archive was exported, published and independently hash-verified before the next profile. Independent saved-record audits passed 9,114 FP16, 10,972 int8 and 12,537 NF4 checks.

| DEV measure | FP16 | int8 | NF4 |
|---|---:|---:|---:|
| Complete persisted tasks |1/8|1/8|0/8|
| English complete |1/4|1/4|0/4|
| Arabic complete |0/4|0/4|0/4|
| Semantically correct initial plans |8/8|8/8|0/8|
| Episode Mind correct |26/59|14/59|40/64|
| Action errors / calls |42/54|41/54|31/58|
| Unnecessary calls |0|0|17|
| Protocol-rejected proposals |0|0|7|
| Repeated standalone probes correct |3/10|2/10|0/10|
| Global model calls |123|123|132|
| Run wall seconds |234.564|610.834|349.008|
| Warm load seconds |6.268|7.174|5.914|
| Model/state storage bytes |4,258,275,904|2,446,336,576|1,569,141,056|
| Vision subset storage bytes |813,914,176|411,260,992|216,331,840|
| Peak allocated CUDA bytes |4,517,451,776|2,725,797,376|1,839,384,064|
| Peak reserved CUDA bytes |4,674,551,808|3,223,322,624|1,941,962,752|
| CPU peak RSS bytes |9,685,237,760|5,498,396,672|5,541,400,576|
| Cache file-content bytes |4,266,814,970|4,266,814,970|4,266,814,970|

Lower quantized storage did not establish a useful replacement: int8 retained one task success with fewer correct Mind decisions and longer wall time; NF4 lost the sole successful task despite a higher adaptive episode Mind count. Different reached states/prompts, call counts and output lengths prevent interpreting aggregate Mind denominators or wall times as identical-question accuracy or causal throughput estimates. All ten standalone probes repeat and were exposed in DEV; they are diagnostics, not new holdouts. Seven NF4 rejections are protocol failures, not seven confirmed harmful external actions. Its thirteen episode model errors include those seven Type errors plus six Mind errors; do not add overlapping counters.

The runtime checked actual quantized layer types; saved int8/NF4 tensor records show 300 quantized weights including 104 vision weights, with all recorded tensors/state on CUDA. The saved records do not list a complete module-class inventory, independently prove pointer/inode deduplication, or attest truthful hardware capture. Independent replay verified all 166 action/frame occurrences across the three runs; repeated occurrences reuse 18/19/18 unique PNGs. Raw storage, allocator reservation, whole-device samples and CPU RSS describe distinct quantities. The allocated idle second T4 and launcher/notebook processes are outside current-model-process cost. Cache file content is not allocated disk blocks or a monetary price.

Power telemetry remains uncalibrated. Raw maxima were 117.67/74.85/89.59 W, while a later same-UUID query reported a 70 W configured and maximum limit. The cause and earlier limit history are unverified. Preserve the readings, but infer no energy, efficiency or monetary cost from them. Sampled maxima are not continuous peaks. Fixed temporal order and warm disk/OS caches further limit cost comparisons.

## Durable identities

Prospective attempt2 artifact: `6a5ac9c7400be06beec608c3aaecac67277e776f`; experiment SHA256 `60a0986cc196041c2474314624e0f88c9c2986d75bc58cf0ddb44623ed294e02`. Original adapter manifest remains `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d`; no retraining, parser repair or data change occurred. Source freeze is `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b`.

| Raw archive | Immutable artifact commit | SHA256 |
|---|---|---|
| profile-0-fp16-dev-evidence.zip |`e252b7805055fa6a89e1d92b6bba7faa5dbf1efc`|`6d5cfe7c3746a76eaceb8ebc37605d0d98cc69b69ca94765421f081b64d64580`|
| profile-1-int8-dev-evidence.zip |`e3d400424766aad3c5856ba29266e11033392ef5`|`e0dbe2d692c522943b3e0c49c198e540416cbabfa2b19499c9da4a518b8339e5`|
| profile-2-nf4-dev-evidence.zip |`d08f522de433d9da866ec3907161e4a9b326efba`|`84dac2fd6ab44d751cefbe9ce61a958ef44a39aad3b1f09e880c5350bbd87458`|

All paths are under `h22-attempt2/`. Actual audits/reviews, public round trips, power addendum and subsequent session-loss observation are retained at `c7f2e16244910c7b4246749b3186d49fca5ce7b4`. The independently authored auditor's original and stricter quantized-dtype version are preserved with genuine Jev/Graft evidence. These tools and saved-record audits are not independent hardware, generalization or release certification.

## Remaining work

Kaggle stopped the allocation at its inactivity prompt before TEST started. The next session had no experiment files and a different physical GPU. **No attempt2 TEST profile was started.** DEV raw records, trained adapter and earlier H21 DEV/TEST comparison remain durable. No completed DEV profile needs repeating. TEST requires a prospective documented hardware/order amendment and reviewed operator; no cross-allocation matched-cost claim is permitted. Attempt1's earlier unretrieved outcomes remain unknown and separate from these retained attempt2 results.

H23 private serving source passed 255 integrated CPU tests with one skip, independent combined/security review after three repair cycles, genuine Jev/Graft and four exact-head CI jobs at `18ad45deaa890c27c7250681e4bffe8b881e5690`. Its actual remote lifecycle remains unrun. The operational builder hit the account usage limit; its partial helper is unreviewed and cannot run. Resume that existing builder after the normal reset, finish independent gates, then execute the separately prespecified actual serving experiment. No billing or plan change is authorized.

No model/precision adoption, native integration or release qualification follows. Founder controls main/production; foundation PR1 remains unchanged.

Follow-up2026-10-17: ordinary review capacity reset, and the completed H23 operator passed independent delta review and12 CPU tests. A separate TEST-only hardware amendment was published; FP16 TEST completed2/8 and is durably retained with9048 independent checks. The subsequent int8 outcome is unknown after a later multi-hour pause and interactive VM expiry. NF4 and H23 remain unrun. A reviewed durable batch plan is now being prepared under ADR0009. These later observations supersede the temporary capacity/operator block above without changing any DEV result. See [TEST restart report](H22-measured-test-20261017.md).
