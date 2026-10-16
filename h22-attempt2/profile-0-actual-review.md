# H22 attempt2 profile0 FP16 DEV: actual evidence review

Verdict: saved-record integrity PASS. `outputs/h22-attempt2-profile-0-audit.json` records 9114 checks, zero errors and zero warnings. All 54 actual actions were independently replayed against half-open normalized/mirrored geometry; all 54 frame-occurrence state bindings, PNG hashes/signatures/dimensions and frozen PNG byte matches pass. The archive has 25 files, including 18 unique PNGs reused across these occurrences. This is the actual retained-adapter FP16 DEV reference run, not a precision comparison or adoption decision.

The 274574-byte ZIP SHA256 is `6d5cfe7c3746a76eaceb8ebc37605d0d98cc69b69ca94765421f081b64d64580`; extracted result SHA256 is `2536e446322228331b8b3dfd45002c363e8ab4a0144097c16b47ee0c638b61ce`. CRCs and every extracted entry's bytes agree. Exact source `07591f109b6683c9302fcf890259ddf0bde91494`, runtime `20b3bf32ca37107ed05e9f2c3ed74413653b51107f5abf474f4a40b25ef58895`, precision source `1fb95afef7254b73f7229ec0da5e6ad6d2e91b42f42e5b4e0476f56b75acd9f5`, source freeze `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b`, original H19 data/freeze and original adapter manifest `7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d` pass independent byte/identity checks. Fixed profile 0 is FP16 DEV; no fallback mode or source adjustment is inferred.

Independently rescored complete persisted tasks are 1/8: English 1/4, Arabic 0/4. Semantically correct plans 8/8; episode Mind 26/59; action errors 42/54; one environment save failure and zero recoveries; five episode model errors. There are 113 episode calls plus ten repeated standalone diagnostics = 123 global calls. The repeated standalone exact result is 3/10, not a new untouched holdout. These aggregate quality counts equal the earlier H21 retained-adapter DEV counts; this single reference run establishes no quantization quality outcome. No TEST failure analysis or tuning was performed.

| Actual recorded cost measure | Value | Interpretation |
|---|---:|---|
| Run wall seconds |234.564268298|Includes loading, calls and runtime overhead |
| Load seconds |6.267743557|Warm-cache load, not model download time |
| Generation seconds, summed across global calls |215.872863998|Independent arithmetic agrees |
| CPU process seconds |240.006778940|Aggregated CPU time may exceed wall time |
| First global call wall seconds |4.539678662|First call in this fresh model process |
| Remaining 122-call median wall seconds |1.470972204|Different task/component lengths; descriptive |
| Final peak CUDA allocated bytes |4517451776|Allocator high-water measurement |
| Final peak CUDA reserved bytes |4674551808|Allocator reservation high-water measurement |
| Maximum runtime sampled device used bytes |4843241472|Whole-device before/after samples, not continuous peak |
| Full model/state unique-storage bytes |4258275904|Recorded storage sums; distinct from allocator use |
| Vision subset unique-storage bytes |813914176|Vision subset sums independently agree |
| Unique physical cache-content bytes |4266814970|File-content count, not allocated disk blocks or price |
| CPU peak RSS bytes |9685237760|Recorded process high-water RSS, including load |
| Final CPU RSS bytes |3451183104|Current RSS differs from prior peak |

The bounded process observer retained 115 samples. The initial row correctly reports `DeviceIdentityUnavailable`, device=null and UUID=null because identity was not resolved yet; it is not silently treated as zero usage. Subsequent device observations bind to recorded physical UUID `GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1`, as does the comparison identity. Their maximum used value is 4619 MiB, maximum utilization 99%, and maximum sampled power 117.67 W. MiB uses 1048576 bytes; observer MiB samples and runtime byte samples have different sampling times/rounding and must not be conflated into one peak. Sample clocks, null/error explanations, utilization bounds and unit/type checks pass. There is no reported top-level telemetry failure. No energy integral or monetary price is inferred from sampled watts.

The result records Tesla T4, CUDA 12.8 and Python 3.13.15. Exact package/import pins agree: torch 2.11.0+cu128, Transformers 5.16.1, Pillow 11.3.0, PEFT 0.20.0, safetensors 0.8.0, accelerate 1.12.0 and bitsandbytes 0.50.2. Recorded model/state tensors reside on cuda:0; per-tensor storage/vision sums pass. The audit cannot independently establish pointer/inode deduplication or truthful hardware capture from aggregate saved metadata. It imports only the independent stdlib audit helper and no production parser, simulator, verifier, precision code, or ML packages. For unchanged H19 record rescore it temporarily projects only the outer reference schema; original result JSON remains unchanged and SHA-bound.

This run follows an explicitly recorded prewarm and the first position in a fixed temporal order. Faster loading than earlier fresh allocations cannot establish a causal precision or hardware speed advantage. Only GPU0/current-process costs are sampled; an idle allocated second T4 is outside these measurements. The physical UUID supports recorded pairing identity, not independent hardware attestation, and cannot remove warm-cache or temporal-order confounding. Missing telemetry must remain unavailable, and sampled maxima remain unqualified as continuous peaks.

`adoption_qualified`, `quality_qualified`, `native_qualified` and `execution_authorized` remain false in both source result and audit. No paid-cost, equivalence, representative generalization, native execution, or release conclusion follows. Preserve this raw FP16 DEV reference and its verified archive before advancing to the next prespecified profile.
