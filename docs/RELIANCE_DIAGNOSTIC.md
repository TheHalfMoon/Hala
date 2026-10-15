# Actual local decision diagnostics

The existing local d1 service was online and used genuinely. No model was downloaded. The installed artifact is `d1-omni-600M-Q8_0.gguf`, 407,207,584 bytes, SHA-256 `cd94463f4cac9c700ec6750df8df005bb6846de410e159d1f5e985e1171693d3`. This is the existing local quantized stack, not a qualified Hala checkpoint or a reproduction of the publisher's FP16 runtime.

`experiments/reliance/d1-diagnostic-20261014.json` records six authored route cases. Three English cases selected deterministic/mind/type as expected, with tool wall times 781/536/588 ms. All three longer Arabic cases returned HTTP 400 before an answer. The earlier eight-question, seven-way request also returned HTTP 400. Causes are unqualified; these failures must not be described as a demonstrated foundation-model Arabic accuracy rate.

A separate minimal greeting probe on `مرحبا` did return a typed answer, but assigned only 0.06754 to “Arabic greeting.” A cancellation probe returned 0.98360 to the explicit cancelled condition. These are individual diagnostics, not safety approval or calibration evidence. The deterministic cancellation/budget paths remain mandatory regardless of model output.

The runtime is advisory only. H9/E4 is incomplete: no heldout route catalog, end-to-end task/call savings, larger-model routing comparison, or commercial release qualification exists. Default deterministic routing is retained. The local calls have no per-call API charge; electricity and hardware amortization were not measured. Jev used for engineering reviews is separate from Hala's runtime dependencies.
