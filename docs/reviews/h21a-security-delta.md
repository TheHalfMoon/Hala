# H21a independent security delta

PASS for reviewed repairs. Original Windows junction finding is closed; no new actionable security finding in this delta. No source changes, model loading/downloads, online inference or full-suite rerun by reviewer.

Independent bounded probes (`work/h21a_security_delta_probes.py`, exit 0) used the complete synthetic 112-tensor helper as the positive fixture, rather than the old invalid partial adapter:

- Created a real Windows junction, removed the in-process `Path.is_junction` API to reproduce Python 3.11 availability, and independently confirmed rejection both as source and destination parent. The `lstat` reparse-point guard remains effective without that API. The probe restored the API and removed only the junction itself. This used bundled Python 3.12 with API absence simulated, not a separate Python 3.11 interpreter.
- Re-derived all 112 names and shapes from independently hash-pinned local official Qwen configuration (`bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b`). They match 28 layers, hidden2048, query2048/value1024 outputs and rank4 A/B dimensions. Complete synthetic bytes roundtrip successfully with quality qualification false.
- Partial inventory, nonexistent layer99, incorrect shape, alternate prefix alias and incorrect dtype/length metadata reject. Previous acceptance of q-only arbitrary-shape layer99 adapters no longer occurs.
- Exact PEFT0.20 producer regex and the provided official inert default fixture roundtrip with original configuration bytes preserved. Broader target regex, auto-mapping, PISSA initialization, non-null alternate variants, nonempty runtime settings and wrong PEFT version reject. The expanded defaults expose no arbitrary code loading in this archive tool. Actual PEFT deserialization/model behavior remains the separately required remote gate.

Reviewed source SHA256:

- `training/adapter_archive.py`: `95773e65224b6ee57c89f7edd61a3d7b247cd623eef3cca4637eed7cbc6abc55`
- `tests/test_adapter_archive.py`: `809caf2d551eb7e0991bb5935efee73b4ee88376adff26290c4bdfc4a5b48227`

Trust scope remains unchanged: externally pinned manifest digest/identities, caller-exclusive directories, structural validation only, no quality or trained-asset attestation.
