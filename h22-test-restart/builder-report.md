# H22 remaining TEST-only restart source handoff

No GPU execution, model download, publication, commit or production source edit. Original DEV profiles all completed and have durable raw public ZIPs; none will rerun. The original matched GPU expired before TEST index3 began. The replacement VM observed after expiry is not treated as qualified hardware. Root must qualify the actual newly started UUID and externally freeze the new TEST amendment before inference.

Frozen operator source:

| File | SHA256 |
|---|---|
| `outputs/h22-test-only-operator.py` | `d8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e` |
| `outputs/h22-test-restart-qualify.py` | `16d1c476ab0a5321c54d30838b4b7f96fc069990bc44efab11fedf994622e48a` |
| `outputs/h22-test-restart-prospective-amendment.json` draft | `94b5a33358aa24c0335e5ea709aa707068f39d65c9697a56ca0c547b1a8e5f2f` |

The amendment GPU field is deliberately `PENDING_ACTUAL_QUALIFICATION`, which the operator rejects. Root replaces it with the actual newly qualified UUID and publishes the resulting immutable artifact commit/SHA. The amendment schema is `hala.h22.test_restart_amendment.v1`, attempt3, with only `mode_order=[['fp16','test'],['int8','test'],['nf4','test']]`. Local indices0..2 are the original study's logical slots3..5. Runtime07591, freeze98ca, source/data/font/model/adapter/pins/budgets remain identical; no parser/task/training change or selection based on new TEST outcomes.

Original DEV group is explicitly `60a0986cc196041c2474314624e0f88c9c2986d75bc58cf0ddb44623ed294e02`. `dev_retention` fixes all3 original index/commit/receipt/archive hashes in the amendment/helper. Public commits are fp16 `e252b7805055fa6a89e1d92b6bba7faa5dbf1efc`, int8 `e3d400424766aad3c5856ba29266e11033392ef5`, nf4 `d08f522de433d9da866ec3907161e4a9b326efba`. First TEST index0 fetches/verifies every old DEV receipt and actual raw ZIP using its original group SHA and complete mandatory raw provenance/result checks before creating output or launching a process. The confirmed receipts are retained in its new evidence ZIP. Missing old DEV evidence blocks TEST.

Qualification command on genuinely restored remote source:

```text
python h22-test-restart-qualify.py --expected-gpu-uuid <externally-observed-new-actual-UUID>
```

This preserves the old qualifier untouched. The new qualifier enforces clean07591 before imports, exact packages/font/H19/retained archive before ML, genuine provider, actual UUID and bounded sampler. It atomically creates `/kaggle/working/hala-h21/h22-test-restart-qualification.json` without overwriting. No prior-VM status or fake GPU metadata is used. A separate `h22-test-restart-prewarm.json` may record original-base cache preparation; qualification does not prewarm or claim a warm cache.

One-profile invocation:

```text
python h22-test-only-operator.py --index 0 --manifest-commit <40hex-public-amendment-commit> --manifest-sha <new-external-TEST-group-SHA256>
```

Manifest URL is fixed to `https://raw.githubusercontent.com/TheHalfMoon/Hala/<40hex>/h22-test-restart/experiment.json`. Each invocation launches exactly one fresh subprocess with3700-second timeout; no DEV action, no automatic advancement, and no existing profile directory/ZIP overwrite. Local output lives at `/kaggle/working/hala-h21/precision-test-restart/profile-N-MODE-test/`, atomic `profile-status.json` inside, and sibling `profile-N-MODE-test-evidence.zip`. Status schema remains `hala.h22.single_profile.v1` and binds local index/name/new experiment SHA/source/argv/raw result status/hash.

Later TEST profiles require the immediately prior new TEST receipt and actual raw ZIP, using `--previous-commit`, `--receipt-sha`, `--archive-sha`. Public paths are exactly `h22-test-restart/profile-N-MODE-test-evidence.zip` and `h22-test-restart/profile-N-MODE-test-receipt.json`. Receipt schema remains `hala.h22.public_retention_receipt.v1` with local index0..2, new `experiment_sha256`, actual `archive_sha256` and exact archive path. Mandatory raw log/qualification/launcher, prior index/mode/split and raw claimed result hash/status/source/freeze are checked; successful measured claims require actual measured result and matching new GPU. Failures/timeouts remain retainable terminal evidence, never qualification claims.

H23 coordination: the SHA-pinned helper can be loaded from `/kaggle/working/h22-test-only-operator.py`; `prior_retention(index+1,test_experiment_sha256,artifact_commit,receipt_sha,archive_sha)` defaults to the fixed TEST group. `group='dev'` selects only original DEV order/prefix/attempt2 and additionally requires the original60a... SHA. H23's table should have exactly two frozen groups with `dev_experiment_sha256` and separately bound `test_experiment_sha256`,3 rows per group. Original DEV checks are public-only because old VM files vanished; new TEST checks additionally bind current local terminal evidence. No cross-group same-physical-GPU or cost equivalence claim is permitted.

Validation:16 stdlib operator tests passed in0.400s (9 preserved attempt2 tests plus7 TEST-restart tests); both new scripts compile. New tests check TEST-only order, missing/changed amendment identity and GPU placeholder, source rejection before fetch/imports, all3 fixed DEV prerequisite calls, actual locally retained DEV ZIP SHA/content compatibility, missing DEV/wrong GPU blocking before output/process, forged/missing raw result evidence, and one mocked TEST-only process/no rerun with archived DEV receipts. These are CPU operator tests, not new model or device evidence. Original operator remains SHA `3e24085ce05b16a0e511e6b03f4fd2dd4138ae0213d5ac23349b6687232e0adc`; original prospective manifest remains SHA `3742fc46ec9f4e5151e24f20a78450ab315c25aaae70b002408a64af44a87f39`.
