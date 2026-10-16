# H22 attempt2 operator source handoff

No inference, GPU execution, downloads, publication or commit by this builder. Production source07591, frozen H19/model/adapter/budgets and the original prospective manifest are unchanged. Attempt1's ephemeral VM disappeared after the user pause; zero precision results were retrieved, so there is no attempt1 quality/cost conclusion.

Files:

- `outputs/h22-attempt2-qualify.py`: frozen SHA256 `1ef101eb2ca16305c06f6739ce00138b8b0b75236e18316775b072805e77d373`. Checks complete clean07591 Git tree before source imports and again before ML imports, exact original packages including Torch2.11.0+cu128, genuine provider, frozen bundle/font and privately verified retained adapter. Requires externally observed new GPU UUID, exercises UUID-bound/bounded sampler, and atomically writes `h22-attempt2-qualification.json` without overwriting. No H21 previous-status dependency, Git mutation or model prewarm. Cache observation explicitly precedes prewarm.
- `outputs/h22-run-one-profile.py`: repaired/frozen SHA256 `3e24085ce05b16a0e511e6b03f4fd2dd4138ae0213d5ac23349b6687232e0adc`. Fixed indices0..5 remain fp16-dev/int8-dev/nf4-dev/fp16-test/int8-test/nf4-test. One profile per invocation,3700-second timeout, fresh subprocess, complete clean07591 source guard before every invocation, no automatic advancement and no overwrite/rerun of existing profile directory. Preserves terminal status, raw log, experiment, qualification, optional separate prewarm record and raw runtime output in evidence ZIP. Failures/timeouts are retained rather than becoming quality conclusions.
- `outputs/h22-attempt2-prospective-experiment.json`: draft SHA256 `2388e1ff4fd971c7e7b0305a9fbf5dc0c46c6fded59b63fe1ea3ea1553e8f48b`. Copies original load-bearing study fields verbatim and changes attempt/hardware/retention/cache metadata only. New UUID is `GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1`. Root reviews/publishes this manifest and supplies its actual immutable artifact commit and SHA; a later manifest edit needs a new external SHA.
- `outputs/test_h22_attempt2_operator.py`: nine stdlib adversarial tests pass in0.289s. Covers wrong/dirty source, changed fixed study fields, invalid indices, mutable URL rejection, redirects/oversize/wrong digest, receipt/archive/current-experiment/previous-index mismatches, missing retention blocking before launch, failure/timeout evidence ZIP, no auto-advance and no rerun. Independent review's missing-raw-evidence/forged-result criteria first reproduced six failing subcases, then passed after the retention-only repair. Measured result tests additionally reject wrong raw status, mode, allocation, source and GPU. ML/network/subprocess execution is mocked; these tests establish no GPU quality.

Qualification invocation:

```text
python h22-attempt2-qualify.py --expected-gpu-uuid GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1
```

Then separately record any original-base prewarm in `/kaggle/working/hala-h21/h22-attempt2-prewarm.json`. Qualification does not claim a warm cache. Actual first run uses:

```text
python h22-run-one-profile.py --index 0 --manifest-commit <40hex-artifact-commit> --manifest-sha <external-experiment-sha256>
```

The manifest is fetched only at `https://raw.githubusercontent.com/TheHalfMoon/Hala/<40hex>/h22-attempt2/experiment.json`, bounded64KiB and checked against supplied SHA. Its fixed model/source/freeze/data/adapter/package/budget/order fields must match the original SHA-pinned publicly retained prospective manifest. Qualification source/freeze/new UUID must match it.

After each invocation, root retrieves the raw ZIP before allowing another profile. Publish it at `h22-attempt2/profile-N-MODE-SPLIT-evidence.zip` and a receipt at `h22-attempt2/profile-N-MODE-SPLIT-receipt.json`, on an immutable40hex artifact commit. Receipt schema:

```json
{"schema":"hala.h22.public_retention_receipt.v1","index":0,"experiment_sha256":"<current-experiment-sha256>","archive_sha256":"<actual-raw-zip-sha256>","archive_path":"h22-attempt2/profile-0-fp16-dev-evidence.zip"}
```

For the next fixed index, pass `--previous-commit <receipt-and-zip-artifact-commit> --receipt-sha <external-receipt-sha256> --archive-sha <external-raw-zip-sha256>`. The launcher bounds/fetches/verifies both actual public receipt and raw ZIP bytes, then checks the ZIP's experiment/status binding to exactly the previous index and current experiment. Local existence or a receipt alone is insufficient. Terminal failed/timeout profiles may satisfy retention; independent next profiles remain prespecified, with no fallback or rerun.

The repaired gate requires actual raw log, qualification and nonempty operator-launcher members alongside experiment/status; it CRC-reads the raw log and binds qualification to the current source/freeze/attempt/GPU. Raw status must bind the exact prior integer index, profile name, source and mode/allocation arguments. Any result/status/hash claim or retained result requires actual `run/result.json`, exact SHA256, matching raw status and fixed source/freeze/mode/allocation. Successful completion additionally requires returncode0, raw measured status and matching prospective GPU UUID. Metadata-only or forged-result ZIPs cannot advance the sequence. Qualification and prospective-manifest bytes were not changed by this repair.

Limits: network reads have45-second timeout/explicit byte caps; prior ZIP at most16MiB,1000 unique members/64MiB expanded total/eightMiB per member. Generated evidence file overflow leaves raw disk evidence and fails packaging for manual retention rather than silently truncating it. Atomic JSON flushes/fsyncs before replacement. Source guards use30-second Git timeouts. None of these operations restore lost prior VM results or establish native/production/Arabic/precision adoption qualification.
