# H22 attempt2 profile1 int8 DEV: actual evidence review

Verdict: saved-record integrity PASS. `outputs/h22-attempt2-profile-1-audit.json` records 10972 checks, zero errors and zero warnings. All 54 actual actions and 54 frame-occurrence bindings/hashes/geometry/state transitions pass independent replay. The 27-file archive contains 19 unique PNGs; all referenced occurrences match frozen external PNG bytes. No local RAQM rerender or GPU inference was performed.

The 294412-byte ZIP SHA256 is `e0dbe2d692c522943b3e0c49c198e540416cbabfa2b19499c9da4a518b8339e5`; extracted result SHA256 is `0f055bce8e2e15a6276f8e30bbc09980c3388d9a142265f71cf1ad5687a51b60`. CRCs, extracted bytes, fixed profile index 1/int8/DEV, original retained adapter, data, exact package/import pins, historical source `07591f109b6683c9302fcf890259ddf0bde91494`, runtime/precision/source-freeze hashes and physical UUID `GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1` all agree. Its bound comparison identity exactly matches the saved FP16 DEV reference. The archived previous-retention receipt identifies profile 0 and its verified ZIP hash; profile status records successful process exit and the same actual result hash.

| Actual quality count | FP16 DEV | int8 DEV |
|---|---:|---:|
| Complete persisted tasks |1/8|1/8|
| English complete |1/4|1/4|
| Arabic complete |0/4|0/4|
| Semantically correct initial plans |8/8|8/8|
| Episode Mind correct |26/59|14/59|
| English episode Mind correct |8/27|10/27|
| Arabic episode Mind correct |18/32|4/32|
| Action errors |42/54|41/54|
| Environment failures / recoveries |1/0|1/0|
| Episode model errors |5|5|
| Global calls, including ten repeated standalone probes |123|123|

Per-episode completion pairs are identical: the same single English episode succeeds. Equal 1/8 completion does not establish equivalent accuracy or preserved component quality; Mind correctness is lower in this observed int8 run, especially Arabic. Standalone exact score is 2/10 versus FP16 3/10 across previously exposed repeated probes, not a new holdout. No TEST failure analysis, prompt/data change, fallback, training, or precision selection was performed.

The exact reviewed runtime uses the int8 bitsandbytes profile and refuses loading unless an actual `Linear8bitLt` module exists. Saved residency separately contains 300 `torch.int8` weight-tensor records, including 104 vision records, all on cuda:0. The full record has 740 tensor entries: 325 FP16, 300 int8 and 115 F32; the vision subset has 211 FP16, 104 int8 and one F32 entry. These are mixed-dtype model/state records and are not a claim that every tensor, including the unchanged LoRA, is int8. The saved result does not enumerate each module's class, so 300 independently listed module classes are not asserted. Quantized-weight and vision-weight dtype evidence and source guard jointly support the qualified software quantization claim. Storage sums do not independently reconstruct tensor pointers or prove honest hardware capture.

| Actual recorded cost | FP16 DEV | int8 DEV |
|---|---:|---:|
| Wall seconds |234.564268298|610.834425637|
| Load seconds, warm disk cache |6.267743557|7.173520617|
| Generation seconds |215.872863998|591.483870811|
| CPU process seconds |240.006778940|615.964790529|
| First call wall seconds |4.539678662|11.478334676|
| Other 122 calls median wall seconds |1.470972204|3.908879571|
| Final peak CUDA allocated bytes |4517451776|2725797376|
| Final peak CUDA reserved bytes |4674551808|3223322624|
| Maximum runtime sampled device used bytes |4843241472|3425566720|
| Recorded full model/state unique-storage bytes |4258275904|2446336576|
| Recorded vision unique-storage bytes |813914176|411260992|
| Unique cache-content bytes |4266814970|4266814970|
| CPU peak RSS bytes |9685237760|5498396672|

Descriptively, int8 records 42.55% less model/state storage and 39.66% less peak allocated CUDA memory, while total wall time is 2.60× FP16 and summed generation time 2.74×. These measurements have different scopes: model storage, allocator peaks and sampled whole-device use cannot be substituted for one another. Fixed temporal order, shared warm disk cache/variable OS page cache, adaptive generation and one run per profile prevent a general causal latency or cost conclusion. No monetary price is inferred.

The observer has 300 bounded samples. The initial null UUID/device row explicitly reports `DeviceIdentityUnavailable`; subsequent device samples bind to the expected physical UUID. Sample maxima are 3267 MiB used, 87% utilization and 74.85 W power. MiB uses 1048576 bytes; sampled watts are neither energy nor price. These whole-GPU/current-process observations exclude cost accounting for an idle allocated second GPU, and are not continuous peaks. Null/error explanations, clocks, byte arithmetic and storage/vision sums pass; no top-level telemetry failure is reported.

Audit revision note: the first run with the already retained auditor passed 10970 checks. The new auditor adds exactly two stricter checks for quantized modes: (1) positive recorded quantized-storage evidence of `torch.int8`/`torch.uint8` for int8/NF4 respectively, and (2) positive vision-subset quantized-storage evidence of that dtype. Re-running int8 passes 10972 checks. No check was weakened and original immutable auditor/source evidence must remain preserved. Current auditor SHA256 is `a816cde40e58b361f7143e5adb165006a877b5a9c39ef9b88540a59e05d1a09f`; it requires separate retention/gates as directed by the parent.

Adoption, quality, native and execution qualification flags remain false in source result and independent audit. This verifies bounded saved evidence and a descriptive DEV comparison; it does not authorize switching precision, adoption, release or a broader capability claim.
