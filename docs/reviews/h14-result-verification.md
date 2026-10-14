# Independent H14 saved-result verification

Offline verification only: no model rerun, tuning, project imports, frozen-source edits, or commits. The independently implemented parser applies the frozen exact schema, explicit JSON fence, integer normalized mapping, half-open rectangles, one selection event, and terminal-state comparison.

Result SHA-256: `6a1adf9a4dd322c06e07ed317f1d880763c60c214d1cc87c036dea21567a7a32`.
Manifest SHA-256: `0d81cc61ecaafb79b52b6a0368f76d550c92f5e4c5ccb05ef468e92a83c5b8d6`.
All 10 frozen source/specification hashes match; result manifest and canonical embedded image-freeze hashes match. All 64 unique task IDs appear exactly once in frozen order, with 32 English and 32 Arabic and four variants per family/language.

Discrepancies: None in saved per-case scores, pixel mappings, summaries, identity/image records, source hashes, memory arithmetic, and timing checks.

Limits: image PNGs, RGB bytes and remote font bytes are not present locally, so their actual contents cannot be rehashed here; embedded provenance consistency is verified. A saved JSON alone cannot independently attest physical GPU execution or freeze chronology. The frozen runner source records actual CUDA operations and renders before model loading; its saved metadata reports Tesla T4, FP16/SDPA, torch 2.11.0+cu130, transformers 5.16.1, Pillow 11.3.0, RAQM 0.10.1, seed 42 and zero training runs.

Eight authored families have dependent variants. These are narrow synthetic one-click selection tasks; geometric target hit and executable fixture completion coincide by design. They are not native Windows/browser execution or general full-task qualification. Wilson intervals below are descriptive only.

| Group | N | Valid | Hit / completion | Wrong target | Blank miss | Invalid | Hit Wilson 95% |
|---|---:|---:|---:|---:|---:|---:|---|
| overall | 64 | 64 | 30 (46.88%) | 19 | 15 | 0 | 35.18%–58.93% |
| en | 32 | 32 | 16 (50.00%) | 11 | 5 | 0 | 33.63%–66.37% |
| ar | 32 | 32 | 14 (43.75%) | 8 | 10 | 0 | 28.17%–60.67% |
| calendar_dates | 8 | 8 | 7 (87.50%) | 0 | 1 | 0 | 52.91%–97.76% |
| chart_peaks | 8 | 8 | 3 (37.50%) | 1 | 4 | 0 | 13.68%–69.43% |
| document_versions | 8 | 8 | 2 (25.00%) | 3 | 3 | 0 | 7.15%–59.07% |
| kanban_priority | 8 | 8 | 3 (37.50%) | 5 | 0 | 0 | 13.68%–69.43% |
| palette_colors | 8 | 8 | 5 (62.50%) | 0 | 3 | 0 | 30.57%–86.32% |
| seat_locations | 8 | 8 | 7 (87.50%) | 0 | 1 | 0 | 52.91%–97.76% |
| transit_arrivals | 8 | 8 | 1 (12.50%) | 7 | 0 | 0 | 2.24%–47.09% |
| warehouse_stock | 8 | 8 | 2 (25.00%) | 3 | 3 | 0 | 7.15%–59.07% |

All stored Wilson bounds, rates and counts for all five metrics also match exact recomputation.

Latency in seconds; linear interpolated quantiles at `(n-1)*q`. Call time includes preprocessing, inference, decoding and scoring, excluding rendering/model load.

| Set / metric | Min | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| all64 / preprocessing_seconds | 0.016658 | 0.024851 | 0.022942 | 0.030558 | 0.032664 | 0.143575 |
| all64 / inference_seconds | 1.220892 | 1.468919 | 1.360724 | 1.521969 | 1.925346 | 5.201857 |
| all64 / call_seconds | 1.237995 | 1.494025 | 1.384517 | 1.556780 | 1.948511 | 5.345756 |
| warm63 / preprocessing_seconds | 0.016658 | 0.022967 | 0.022930 | 0.028596 | 0.032537 | 0.044403 |
| warm63 / inference_seconds | 1.220892 | 1.409666 | 1.360564 | 1.518401 | 1.876540 | 2.100638 |
| warm63 / call_seconds | 1.237995 | 1.432886 | 1.383885 | 1.553751 | 1.897197 | 2.133567 |
| calendar_dates / call | 1.237995 | 1.883121 | 1.372804 | 2.684007 | 4.014882 | 5.345756 |
| chart_peaks / call | 1.275091 | 1.396514 | 1.382860 | 1.535027 | 1.546081 | 1.557135 |
| document_versions / call | 1.282020 | 1.696806 | 1.721836 | 2.026077 | 2.079822 | 2.133567 |
| kanban_priority / call | 1.284253 | 1.395510 | 1.377780 | 1.490524 | 1.491132 | 1.491740 |
| palette_colors / call | 1.277596 | 1.392106 | 1.355987 | 1.508889 | 1.543357 | 1.577825 |
| seat_locations / call | 1.273544 | 1.389497 | 1.378176 | 1.513530 | 1.534742 | 1.555953 |
| transit_arrivals / call | 1.261652 | 1.382513 | 1.380288 | 1.496898 | 1.498310 | 1.499723 |
| warehouse_stock / call | 1.271188 | 1.416130 | 1.390400 | 1.512369 | 1.528657 | 1.544945 |

Cold first call: preprocessing 0.143575s; inference 5.201857s; total 5.345756s.
Download/load 89.112903s; total wall 197.181107s; process CPU 137.582827s. CPU seconds are not inference latency or an inference-cost estimate.

| Memory measure | Bytes | GiB (2^30) |
|---|---:|---:|
| loaded allocator allocated | 4256128512 | 3.963829 |
| loaded device used | 4388159488 | 4.086792 |
| maximum allocator peak allocated | 4587264512 | 4.272223 |
| maximum allocator peak reserved | 4624220160 | 4.306641 |
| maximum recorded device used | 4792909824 | 4.463745 |
| final allocator allocated | 4290732032 | 3.996056 |
| final device used | 4792909824 | 4.463745 |
| device total | 15637086208 | 14.563171 |

Device used includes driver/other-process allocations. Allocator peaks are cumulative since the post-before-load reset; per-case peak values are not independent case residency peaks.

| Case | Pixel | Valid | Hit | Wrong | Miss | Completion |
|---|---|---:|---:|---:|---:|---:|
| h14-calendar_dates-00 | [80, 120] | 1 | 0 | 0 | 1 | 0 |
| h14-calendar_dates-01 | [564, 400] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-02 | [860, 192] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-03 | [220, 270] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-04 | [470, 292] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-05 | [326, 183] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-06 | [374, 302] | 1 | 1 | 0 | 0 | 1 |
| h14-calendar_dates-07 | [704, 213] | 1 | 1 | 0 | 0 | 1 |
| h14-transit_arrivals-00 | [400, 180] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-01 | [368, 245] | 1 | 1 | 0 | 0 | 1 |
| h14-transit_arrivals-02 | [384, 480] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-03 | [440, 360] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-04 | [547, 404] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-05 | [480, 384] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-06 | [512, 460] | 1 | 0 | 1 | 0 | 0 |
| h14-transit_arrivals-07 | [440, 432] | 1 | 0 | 1 | 0 | 0 |
| h14-warehouse_stock-00 | [266, 299] | 1 | 0 | 1 | 0 | 0 |
| h14-warehouse_stock-01 | [319, 426] | 1 | 1 | 0 | 0 | 1 |
| h14-warehouse_stock-02 | [256, 433] | 1 | 1 | 0 | 0 | 1 |
| h14-warehouse_stock-03 | [528, 432] | 1 | 0 | 1 | 0 | 0 |
| h14-warehouse_stock-04 | [400, 360] | 1 | 0 | 0 | 1 | 0 |
| h14-warehouse_stock-05 | [480, 446] | 1 | 0 | 0 | 1 | 0 |
| h14-warehouse_stock-06 | [340, 460] | 1 | 0 | 1 | 0 | 0 |
| h14-warehouse_stock-07 | [440, 432] | 1 | 0 | 0 | 1 | 0 |
| h14-kanban_priority-00 | [400, 180] | 1 | 0 | 1 | 0 | 0 |
| h14-kanban_priority-01 | [757, 412] | 1 | 0 | 1 | 0 | 0 |
| h14-kanban_priority-02 | [807, 384] | 1 | 0 | 1 | 0 | 0 |
| h14-kanban_priority-03 | [704, 421] | 1 | 1 | 0 | 0 | 1 |
| h14-kanban_priority-04 | [400, 360] | 1 | 1 | 0 | 0 | 1 |
| h14-kanban_priority-05 | [432, 243] | 1 | 1 | 0 | 0 | 1 |
| h14-kanban_priority-06 | [460, 414] | 1 | 0 | 1 | 0 | 0 |
| h14-kanban_priority-07 | [352, 432] | 1 | 0 | 1 | 0 | 0 |
| h14-seat_locations-00 | [400, 240] | 1 | 0 | 0 | 1 | 0 |
| h14-seat_locations-01 | [757, 431] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-02 | [169, 394] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-03 | [341, 299] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-04 | [466, 299] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-05 | [167, 399] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-06 | [807, 402] | 1 | 1 | 0 | 0 | 1 |
| h14-seat_locations-07 | [682, 198] | 1 | 1 | 0 | 0 | 1 |
| h14-chart_peaks-00 | [640, 300] | 1 | 1 | 0 | 0 | 1 |
| h14-chart_peaks-01 | [96, 192] | 1 | 0 | 1 | 0 | 0 |
| h14-chart_peaks-02 | [348, 460] | 1 | 0 | 0 | 1 | 0 |
| h14-chart_peaks-03 | [352, 432] | 1 | 0 | 0 | 1 | 0 |
| h14-chart_peaks-04 | [80, 180] | 1 | 0 | 0 | 1 | 0 |
| h14-chart_peaks-05 | [286, 437] | 1 | 1 | 0 | 0 | 1 |
| h14-chart_peaks-06 | [102, 230] | 1 | 0 | 0 | 1 | 0 |
| h14-chart_peaks-07 | [396, 432] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-00 | [399, 412] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-01 | [720, 192] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-02 | [807, 384] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-03 | [88, 360] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-04 | [550, 60] | 1 | 0 | 0 | 1 | 0 |
| h14-palette_colors-05 | [959, 64] | 1 | 0 | 0 | 1 | 0 |
| h14-palette_colors-06 | [169, 371] | 1 | 1 | 0 | 0 | 1 |
| h14-palette_colors-07 | [165, 72] | 1 | 0 | 0 | 1 | 0 |
| h14-document_versions-00 | [550, 506] | 1 | 0 | 1 | 0 | 0 |
| h14-document_versions-01 | [660, 440] | 1 | 0 | 0 | 1 | 0 |
| h14-document_versions-02 | [768, 672] | 1 | 1 | 0 | 0 | 1 |
| h14-document_versions-03 | [440, 630] | 1 | 0 | 1 | 0 | 0 |
| h14-document_versions-04 | [470, 273] | 1 | 0 | 0 | 1 | 0 |
| h14-document_versions-05 | [557, 439] | 1 | 0 | 0 | 1 | 0 |
| h14-document_versions-06 | [594, 652] | 1 | 0 | 1 | 0 | 0 |
| h14-document_versions-07 | [440, 360] | 1 | 1 | 0 | 0 | 1 |
