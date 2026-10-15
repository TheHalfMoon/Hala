# H14 actual frozen FP16 run

On 2026-10-14 the unchanged pinned Qwen3-VL-2B-Instruct ran once on all 64 frozen synthetic selection fixtures on a free Colab Tesla T4. The source checkout was `ae0a9977edb2ec51825a8d5be5c2a3bdf9eb61ae`. Manifest SHA-256: `0d81cc61ecaafb79b52b6a0368f76d550c92f5e4c5ccb05ef468e92a83c5b8d6`. Exact measured JSON was copied from the rendered Colab output; independent offline recomputation found no score/hash discrepancies. Original screenshots remain ephemeral in the remote runtime; local recomputation cannot independently attest GPU execution or physical rendering chronology.

| Measure | Result |
|---|---|
| Schema validity | 64/64 |
| Target hit / one-click synthetic completion | 30/64 (46.875%), descriptive Wilson 95% interval 35.176–58.928% |
| Wrong target | 19/64 (29.6875%) |
| Blank miss | 15/64 (23.4375%) |
| English / Arabic hit | 16/32 (50%) / 14/32 (43.75%) |
| Warm call median / P95 | 1.383885s / 1.897197s |
| Cold first call | 5.345756s |
| Peak allocator allocated / reserved | 4,587,264,512 / 4,624,220,160 bytes |
| Maximum total device used | 4,792,909,824 bytes; includes driver/other process residency |
| Whole run wall time | 197.181107s, includes cold download/load |
| Resource | Free Colab T4; no paid action activated |

Family successes out of eight: calendar 7, seat 7, palette 5, chart 3, kanban 3, document 2, warehouse 2, transit 1. These dependent authored variants are neither random samples nor native desktop tasks. Confidence intervals are descriptive only. No protocol, model or task tuning followed these test outcomes. The weak results motivate independent future training data, not training on these held-out cases. The original strict 0/8 and same-fixture normalized calibration 7/8 records remain preserved.

This run establishes a measured experimental baseline, not representative model, native task, serving or release qualification. No learned adapter was used.
