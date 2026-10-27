# H159 — deterministic original-application screenshot admission

H158's model evaluation was invalidated because 8/16 original TodoMVC image recreations failed exact SHA identity. The previous 16 samples and 32 GPU predictions remain quarantined and will not be rerun, repaired, tuned, or scored.

H159 is a NEW, balanced 16-scene source engineering cohort (original TodoMVC indices 301–316, 8 English / 8 Arabic, three task families) using the same pinned unmodified MIT upstream. It freezes screenshots **after DOM-state confirmation**, waits for fonts, disables animation during capture, and requires two identical PNG byte sequences before saving. It then recreates the exact scenario in a NEW browser context and demands 16/16 full PNG SHA matches before any model use. It excludes all 16 previous H158 image SHA values.

Commands on owner Windows (no model/GPU):
```powershell
cd C:\Users\Shehr\work\Hala\h154-isolated
python -m unittest tests.test_h159_stable_source_identity
python -m h159.stable_source_qa capture --owner C:\Users\Shehr\work\Hala --upstream C:\Users\Shehr\work\Hala\private-h158\upstream\todomvc
python -m h159.stable_source_qa verify --owner C:\Users\Shehr\work\Hala --upstream C:\Users\Shehr\work\Hala\private-h158\upstream\todomvc
```

All original source and screenshot data stays owner-private. This phase is source quality only, not a neural performance improvement, and does not change flagship Hala 35B. No H140 sealed blind, paid GPU, release, merge or model publication. If exact replay is not achieved, stop and create a new unseen data cohort instead of changing scores on consumed samples.
