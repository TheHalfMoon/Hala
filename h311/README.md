# H311 — exact familywise paired-GUI inference guard (hypothetical only)

**No Qwen inference, new training, private benchmark, scored OOD, or model gain.**
This addresses a prospective scientific design gap after H310: a future
multi-family paired evaluation cannot select the most favorable source
family, pool unlike source families into one p-value, or treat a significant
negative direction as an adapted-model gain.

`h311.familywise_paired_evidence_guard` accepts only strictly structured
**invented** family-level discordant counts. It computes each family's
exact two-sided conditional McNemar p-value via H310, applies Holm's
step-down adjustment across the complete declared set (2..8 families),
and requires a positive adapted direction **in every declared family**
before returning a hypothetical all-family success flag. Ties are
deterministic; missing/duplicate/reordered families, bool counts,
extra fields and oversized discordance fail closed. No file or CLI
reader exists; no model outputs, screenshots, labels, private aggregates,
task text, points, weights, or historical OOD are accessed or emitted.

Holm controls familywise type-I error under arbitrary *between-family*
dependence **if the individual p-values are valid**. Exact conditional
McNemar still assumes valid paired sampling *within* each family. A
software function cannot prove independence, preregistration, legal
source rights, unbiased sampling, blind custody, or adapted benefit.

For an invented two-family illustration with BASE-only=0 in each family
and ADAPTED-only=8 and 7, exact p-values are 1/128 and 1/64;
Holm-adjusted p-values are 1/64 for each. This is NOT an observation.
The historical H292 exposed BASE6/8 vs ADAPTED6/8 has **zero gain**,
and no authenticated case-level pairing is supplied to H311.

Run source-only tests on Python 3.11/3.12:

```bash
python -m unittest discover -s tests -p test_h311_familywise_paired_evidence_guard.py -v
python -O -m unittest discover -s tests -p test_h311_familywise_paired_evidence_guard.py -v
python -m compileall -q h311
```

**Release remains prohibited.** H199 official admitted TRAIN manifest is
EMPTY. H173 scored OOD is consumed/sealed; H212 scored rows barred;
H250/H251 external media quarantined. H293 output access and H294
provenance write safety blocks are untouched. All ShiftDesk/FieldPort/
Queueboard/RouteBoard/ClinicGrid sources are public/exposed. A real
future study needs independently reviewed lawful diverse TRAIN/DEV,
new never-exposed separately custodied source-disjoint blind OOD,
prospective family-aware paired base-vs-adapted native GUI gains with
uncertainty and safety/abstention, not merely this hypothetical tool.
