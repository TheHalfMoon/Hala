# H22 attempt2 profile2 NF4 DEV: actual evidence review

Verdict: saved-record integrity PASS; task quality remains unqualified. `outputs/h22-attempt2-profile-2-audit.json` records 12537 checks, zero errors and zero warnings using unchanged auditor SHA256 `a816cde40e58b361f7143e5adb165006a877b5a9c39ef9b88540a59e05d1a09f`. All 58 actual actions, complete before/after/observe/final-state chains, half-open mirrored target geometry, flags and frame-occurrence bindings pass independent replay. The 26-file archive contains 18 unique PNGs reused across these actions; all 58 referenced occurrences match frozen PNG bytes. No independent local RAQM rerender or model execution is claimed.

The 281137-byte ZIP SHA256 is `84dac2fd6ab44d751cefbe9ce61a958ef44a39aad3b1f09e880c5350bbd87458`; result SHA256 is `15c2c48f119070b75a0e3fe3f816c409e34109141023f05894f2ec738c9a44db`. CRCs/extracted bytes, original retained adapter/data, fixed index 2/NF4/DEV, source `07591f109b6683c9302fcf890259ddf0bde91494`, immutable runtime/precision/source freeze and package pins all agree. Comparison identity exactly matches FP16 DEV, including recorded physical UUID `GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1`. Archived prior-retention receipt binds profile 1's verified int8 ZIP; profile status reports exit 0 and the same result identity. Frozen inputs, strict protocol and scoring remain unchanged.

| Actual DEV count | FP16 | int8 | NF4 |
|---|---:|---:|---:|
| Complete persisted tasks |1/8|1/8|0/8|
| English / Arabic complete |1/4;0/4|1/4;0/4|0/4;0/4|
| Semantically correct initial plans |8/8|8/8|0/8|
| Episode Mind correct |26/59|14/59|40/64|
| Action errors |42/54|41/54|31/58|
| Unnecessary action calls |0|0|17|
| Protocol-rejected visual proposals |0|0|7|
| Episode model errors |5|5|13|
| Environment failures / recoveries |1/0|1/0|0/0|
| Global calls including ten repeated standalone probes |123|123|132|

NF4 loses the sole English episode completed by FP16 and int8; the other seven completion pairs remain failures. Its 40/64 Mind score (English 14/32; Arabic 26/32) is evaluated over reached, response-dependent states and a different denominator than the references. It does not establish improved fixed-question accuracy or compensate for zero completed tasks. Initial plan correctness is semantic, not a claim that every initial plan is invalid JSON. All 17 unnecessary calls are actual visual action-trace counters, with no redundant-observation contribution. Thirteen episode model errors comprise seven Type protocol rejections and six Mind errors; these overlap other error counts and must not be added as independent events. The seven `unsafe_proposals` are rejected protocol proposals, not independently confirmed harmful/native actions. Repeated standalone exact score is 0/10 with four invalid responses; these probes were previously exposed and are not new holdout evidence. No TEST failure analysis or remediation/tuning was performed.

The exact precision loader configures NF4, double quantization and packed uint8 storage with FP16 compute, refuses loading without an actual `Linear4bit` module, and enforces cuda:0 residency. Saved residency has 1940 tensor/state entries: 1015 F32, 600 uint8 and 325 FP16. Of the uint8 entries, 300 are packed weight records, including 104 vision weight records; the other uint8 entries include quantization state. The vision subset has 313 F32, 208 uint8 and 211 FP16 entries. All model/state records are on cuda:0. Independently summed full/vision storage and positive quantized/vision storage checks pass. This supports recorded software quantization with vision included, not a claim that all tensors or the original LoRA are four-bit. Module-class counts are not independently enumerated in the saved result; actual hardware/pointer attestation is outside this audit.

| Actual recorded cost | FP16 DEV | NF4 DEV |
|---|---:|---:|
| Wall seconds |234.564268298|349.007708263|
| Load seconds, warm disk cache |6.267743557|5.914006438|
| Summed generation seconds |215.872863998|330.642020964|
| CPU process seconds |240.006778940|352.593964986|
| First call wall seconds |4.539678662|5.478909607|
| Other-call median wall seconds |1.470972204|1.712518124|
| Final peak allocated CUDA bytes |4517451776|1839384064|
| Final peak reserved CUDA bytes |4674551808|1941962752|
| Maximum runtime sampled device used bytes |4843241472|2110652416|
| Full model/state unique-storage bytes |4258275904|1569141056|
| Vision unique-storage bytes |813914176|216331840|
| Unique physical cache-content bytes |4266814970|4266814970|
| CPU peak RSS bytes |9685237760|5541400576|

Descriptively, NF4 records 63.15% less model/state storage and 59.28% less peak allocated CUDA memory, but 1.49× total wall time compared with FP16. The remaining-call medians include 122 calls for FP16 and 131 for NF4 with differing outputs/context; they are not a matched latency trial. Fixed temporal order, warm shared disk cache/variable OS page cache, adaptive generation and dependent authored tasks prevent general causal efficiency or equivalence claims. Model storage, allocator peaks, whole-device samples, file-content bytes and monetary costs are different measures. Cache bytes do not reconstruct inode deduplication or allocated blocks; tensor storage sums do not reconstruct pointer deduplication.

The bounded observer retained 172 samples; the initial unresolved device identity is explicitly unavailable, subsequent samples bind to the expected UUID, and clocks/null/unit checks pass. Observer maximum used memory is 2013 MiB and utilization 100%. Its raw maximum power is 89.59 W, also above the later same-UUID 70 W reported power/max limit. Apply `work/h22-power-telemetry-addendum.md`: sensor values are uncalibrated and unsuitable for energy, efficiency or monetary claims; no cause of the discrepancy is established. Numerical audit success does not validate calibrated physics. No top-level telemetry failure is reported, and sampled maxima remain unqualified as continuous peaks. Only GPU0/current-process observations are covered, not costs of an idle allocated second GPU.

This is an independently rescored saved-record comparison of the prespecified DEV profiles. Adoption, quality, native and execution flags remain false. No frozen source/auditor changes, fallback precision, new training run, precision selection or adoption decision occurred in this review. Preserve the raw NF4 archive before advancing to the next fixed profile.
