# H24 HalaMind — prospective remote DEV-only base benchmark

**Status:** CPU/source preparation only. The actual free Kaggle GPU run is NOT claimed until a private Saved Version completes and its raw output passes the separately published verifier. This is not fine-tuning, a trained HalaMind checkpoint, Orcel device integration, HalaType visual grounding, statistical superiority or release qualification.

## Fixed intended setup

- Frozen H24 dataset: SHA256 `391d0dc925790558fd5d58f6c0e6348c7c7984b05891fbbd9ae956c9564572de`.
- **Twenty new DEV examples only**, in two unrelated application families (purchase requisition and classroom attendance), ten English and ten Arabic. Family-isolated train and TEST remain unused; no H22 TEST trajectories or prior model raw answers are training labels.
- Unchanged public `Qwen/Qwen3-VL-2B-Instruct`, exact revision `89644892e4d85e24eaac8bacfd4f463576704203`, FP16, deterministic generation, max 128 output tokens, no LoRA adapter or optimization/tuning.
- Prospective input: HalaMind receives the public user goal and current trusted observation metadata. The expected decision/label, dataset split/ID and protected training labels are not included. The public prompt deliberately projects away synthetic protected distractor annotations; only display roles/names/bounds are retained. Trusted current-frame identity is metadata, never page instructions. Untrusted page text is explicitly data and never overrides user or Orcel authority. Responses are typed `{"kind":"next","tool":"...","subgoal":"..."}`; malformed JSON, extra keys, shell actions, model errors and overlong raw output fail closed.
- Output scope: separate per-case raw model text; language/family/category identity; exact decision and tool accuracy; independently recomputable 20-case English/Arabic summaries; individual latency; GPU allocator bytes, model load duration and exact source/model/data pins. CUDA allocator bytes are **not** full machine residency, calibrated energy or production per-token cost.

## Implemented admission and audit

- `h24/mind_dev.py`: data validation, DEV-only prompt projection, strict response parser, CPU scorer and fail-closed **Kaggle Batch/T4-only** real runner. All actual ML imports and model weight fetches occur only after remote runtime, exact clean Git commit, pinned dependencies and frozen source checks. Explicit remote opt-in required; source/weights are never downloaded locally.
- `tools/build_h24_kaggle.py`: produces a private Kaggle GPU+internet **Save & Run All** notebook/metadata from an immutable 40-hex signed code-review commit. Generator itself allocates no GPU, uses no credentials and cannot overwrite output directories. Kaggle kernel is private; no paid services or key activation.
- `tools/verify_h24_dev.py`: requires externally obtained exact raw JSON result SHA256 and actual reviewed source SHA; verifies all 20 DEV outcomes, order, family split, model/revision, bilingual raw outputs and independently recomputed scores, reported GPU/model load/allocator telemetry and persistent no-adoption flags. Output audit is new-file-only and **cannot attest hardware truth**.
- `tests/test_h24_mind_dev.py`, `tests/test_h24_dev_audit.py`: CPU tests on wrong split, forged model authority, JSON injection/duplicates, nonexistent GPU, mislabeled scores, missing/duplicated results and notebook private metadata. Synthetic oracle fixtures establish scorer behavior **only**, never real base-model quality.

## To run after review

From the verified published source tree:

```bash
python tools/build_h24_kaggle.py --source-commit <EXACT_H24_SIGNED_COMMIT_SHA40> --output <NEW_LOCAL_NOTEBOOK_BUNDLE_DIRECTORY>
```

Confirm notebook/metadata are private, GPU/internet enabled, source SHA exactly pinned, and no paid plan. Submit with existing Kaggle CLI `kaggle kernels push -p <NEW_LOCAL_NOTEBOOK_BUNDLE_DIRECTORY>`. Only the remote Kaggle T4 may fetch model weights. After the Saved Version ends, download `/kaggle/working/hala-h24-mind-dev-output/result.json`; compute its true SHA256 and run:

```bash
python -m tools.verify_h24_dev <DOWNLOADED_RESULT_JSON> --expected-sha256 <OBSERVED_RAW_SHA256> --source-commit <SOURCE_SHA40> --output <NEW_AUDIT_JSON>
```

Keep all raw/error outputs, review exact summary and per-language variance, verify synthetic DEV only, then publish immutable receipts and disclose failures. No repeated adaptive benchmark tuning, test contamination, checkpoint adoption, native Orcel actions or main merge.

**Separate unqualified gates:** fresh-context PStack review, Arabic/RTL semantic review and RAQM-qualified Type media, representative closed-loop Hala model performance, independent security/authority review and H23 actual CUDA cancellation/recovery.
