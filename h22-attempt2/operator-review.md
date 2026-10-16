# H22 attempt2 — independent operator review

Scope: single-profile launcher, qualification helper, separate prewarm helper, prospective attempt2 manifest and seven adversarial operator tests. No inference, GPU execution, downloads, publication or source edits by reviewer. Attempt1 yielded no retrieved results; it supplies no precision quality/cost evidence.

## Must fix

- `outputs/h22-run-one-profile.py`, `prior_retention`: the durable-retention gate checks only the ZIP experiment and status metadata. It does not require a raw log, qualification or launcher, nor require/hash-check `run/result.json` when `result_sha256` claims one exists. A CPU probe accepted a ZIP containing only `experiment.json` and `profile-status.json` with `status=finished`, `result_status=measured` and an arbitrary claimed result hash. Thus the next profile can run without the previous raw inference evidence being publicly retained. Require mandatory provenance/raw-log members, and verify any claimed raw result against its hash/status; a reported successful measured run must include that raw result. Add incomplete/missing/changed-result adversarial cases before proceeding.

## Confirmed clean

- Exactly one selected index0..5 launches one fresh process with the unchanged six-mode order, source07591, freeze98ca7, original H19/retained-adapter identities and original package/budget/model-choice fields. New manifest fields distinguish the restored GPU/cache/retention context; original scientific settings are checked against the original immutable SHA-bound manifest.
- Complete clean source and exact Git HEAD are checked before imports/launch; immutable fixed-repository artifact URLs require lowercase40hex commits, external SHA256 and bounded reads. Receipts bind experiment, prior index and actual public ZIP bytes; missing previous receipt blocks output/process creation. ZIP inventory bounds and no extraction avoid path traversal/decompression expansion beyond the checked limits.
- A new per-profile directory and exclusive ZIP creation prevent overwriting/rerunning an existing local profile. There is no automatic advancement. Failed/timeout subprocesses receive terminal status and retained logs; overflow fails packaging while leaving raw evidence on disk. Rerun prohibition is local to this restored working directory and assumes the trusted operator does not reset it or replay an already retained index in another VM.
- Qualification checks actual package imports, source/freeze, provider, bundle, font, exact retained archive and UUID-bound telemetry without loading a model. The source review cannot freshly establish remote GPU facts.
- Prewarm is a separate original-model frozen-revision `snapshot_download` with non-code asset patterns, no model construction and explicit cache/time/file-hash provenance. It uses the Hub client's normal configured endpoint, whose default is the official Hub; an official-source claim assumes the actual environment has no alternate Hub endpoint override. Fixed warm-cache order and page-cache differences remain limitations.

## Validation and identities

Seven existing stdlib tests independently passed in0.194 seconds. The additional in-memory incomplete-ZIP probe reproduced the must-fix above without network, ML or a profile process. Parsed all three helpers successfully.

- Launcher SHA256: `a63d5168e53dc5f592a6266fcd4aa40a5c63595aeed9206918dd5a8236b065eb`.
- Qualification SHA256: `1ef101eb2ca16305c06f6739ce00138b8b0b75236e18316775b072805e77d373`.
- Prewarm SHA256: `1ee033a0fa6de214bb4a58c98ee70c8f8a08f2459fcd94f4872e5cdb5e19f0c4`.
- Attempt2 manifest SHA256: `2388e1ff4fd971c7e7b0305a9fbf5dc0c46c6fded59b63fe1ea3ea1553e8f48b`.

Combined correctness/parsimony/security operator verdict: repair raw-retention completeness before starting the new sequence. No full H22 source audit or broad ML retest was repeated.

## Retention repair delta — resolved

Reviewed repaired launcher SHA256 `3e24085ce05b16a0e511e6b03f4fd2dd4138ae0213d5ac23349b6687232e0adc`. The prior must-fix is resolved: mandatory raw log/qualification/nonempty launcher members are required; the log is CRC-read; qualification and exact prior profile source/index/name/mode/split identities are bound. Any retained/claimed result must have actual matching SHA256 raw JSON and matching status/profile/source/freeze. Finished outcomes additionally require returncode0, measured raw status and the prospective GPU UUID. The previously accepted metadata-only ZIP now fails.

Independently ran all nine small operator tests: passed in0.201 seconds, including missing raw members, missing/forged result hashes, valid measured retention, and wrong raw status/source/mode/allocation/GPU cases. Recomputed the repaired launcher hash above. Qualification, prewarm and prospective scientific inputs were not changed by this narrow repair. No network, model process, ML import or GPU execution occurred in review.

Remaining findings: none for this delta. Combined operator review now clears the fixed trusted attempt2 workflow, subject to externally publishing the exact reviewed helper/manifest and retaining each actual raw result before advancing. No model-quality or adoption qualification is implied.
