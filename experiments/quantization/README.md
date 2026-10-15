# Actual paired precision comparison

| Path | Valid | Hit / one-click completion | English | Arabic | Wrong target | Warm P50/P95 | Peak allocator | Max sampled device use |
|---|---|---|---|---|---|---|---|---|
| FP16 | 64/64 | 30/64 | 16/32 | 14/32 | 19/64 | 1.384/1.897s | 4.587 GB | 4.793 GB |
| 8-bit | 64/64 | 34/64 | 20/32 | 14/32 | 20/64 | 3.961/5.844s | 2.776 GB | 3.197 GB |
| 4-bit NF4 | 64/64 | 32/64 | 19/32 | 13/32 | 21/64 | 1.554/1.700s | 1.909 GB | 2.140 GB |

Frozen source commit f5f6f34ef51e71b16d8ea5c293161991cb4eba8f; final pre-inference source freeze d9cbb88f567e43325028151d39c1a00e2ae9c9081fdd1556094aedce8e69585c. Historical source/spec files remain frozen at their prospective state; these measured records and independent-verification.json contain the outcomes.

These are eight authored families with dependent variants and narrow synthetic one-click selection tasks, not native application or representative general task qualification. Descriptive Wilson95% hit intervals: FP16 35.176–58.928%, int8 41.072–64.824%, NF4 38.102–61.898%. Int8 gained6/lost2 hits; NF4 gained6/lost4 relative to FP16. Arabic was14/32,14/32,13/32. No equivalence, significance, general improvement or adoption claim is made. Separate GPU allocations, cached loads and execution times confound timing comparisons. Device usage is sampled and includes driver/other context; allocator peaks and enumerated tensor storage measure different things.

Quantization used actual bitsandbytes0.50.2/accelerate1.12.0/Transformers5.16.1/Torch2.11.0+cu130/Pillow11.3.0 on a verified free Colab T4. Each quantized path had300 quantized modules, including104 vision modules; all recorded model/state tensors were on cuda:0. Independent standard-library verification passed 2,048 assertions, rescoring all 192 raw outputs and checking summaries, pairs, source pins, timing and memory arithmetic. Recorded storage deduplication cannot be independently proved without pointer identities. No quantized checkpoint or deployment was published.

