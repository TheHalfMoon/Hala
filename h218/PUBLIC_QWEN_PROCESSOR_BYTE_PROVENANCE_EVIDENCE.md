# H218 — byte-exact provenance of the real offline Qwen3-VL processor

**Research-only Draft, stacked on H217 #238** at exact parent
`ba0811ef27b53dfa373ef99c6a7f8801205089ff`.
Never merge to protected main, train any model, publish weights,
access historical owner media/gold, reuse consumed H173 OOD, or use Mac.

## Grounded gap and source evidence

H216 #237 actually ran the Qwen3VLProcessor on **96/96 original
H212 synthetic GUI screenshots** with exact user+assistant prefix,
identical vision tensors and assistant-only loss. H217 #238 now
requires the exact `transformers.Qwen3VLProcessor` runtime class,
but merely naming a revision folder and checking the class cannot
establish that the **processor/tokenizer assets have not changed**.

On owner Windows 11, the existing (previously cached) publicly
available HF `Qwen/Qwen3-VL-2B-Instruct` snapshot
`89644892e4d85e24eaac8bacfd4f463576704203`
contains exactly **8 public processor and tokenizer files**, with
no model weights present in that snapshot. H218 computed SHA256 over
the actual 8 cached files without reading owner-private content,
without accessing the network and without loading any model.
These exact measured SHA256s and byte counts are hard-coded in
`h218/pinned_qwen_processor_bytes.py`. The public source files
are **not** copied into the Git repo.

## Verified genuine owner Windows evidence

`python3.11 -m h218.pinned_qwen_processor_bytes
--local-processor-snapshot <owner-cached-processor-revision>`

Actual owner device emitted the following **twice**, once normal
and once with `python -O`:

`H218_ALL_EIGHT_PUBLIC_QWEN_PROCESSOR_BYTES_PINNED
files=8 public_bytes=11499725 weights=0 model_calls=0
private_data=0 training=0`

Every SHA has exactly 64 lower-case hex characters, and **all
11,499,725 bytes** were read from only those 8 public assets.

## What this blocks before any Qwen3-VL processor import

- H216's existing preflight now calls
  `verify_pinned_snapshot(root)` before invoking
  `AutoProcessor.from_pretrained`, including
  `local_files_only=True, trust_remote_code=False`.
- Exactly 8 previously observed names, sizes and SHA-256 values
  are required; an altered file with the SAME size is denied.
- All additional/unknown files and directories, symlinks,
  model-weight safetensors, missing files, truncated/grown files,
  source path traversal and different snapshot revisions fail.
- Bounded 1 MiB streaming reads with exact expected size checked
  before and after hashing. No scanning outside the explicitly
  supplied public processor snapshot directory. No user-override
  of production trusted file specification.
- Existing `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  checks remain. The model itself is NEVER loaded.
- Synthetic adversarial tests use 8 freshly created fake small
  files in disposable temp directories; these CI fixtures do not
  establish model correctness or load HF assets.
- Never print the tokenizer contents, owner paths or screenshot
  pixels. Positive status prints only public file count and bytes;
  failure reports a redacted qualification error.

## Independent limitations

This checks **local cached bytes against the pinned prior owner
observation**, not an independently signed upstream publisher
signature. The pinned model revision alone is not a signature
verification or proof that weights from the same repo exist or
are usable. Host filesystem TOCTOU cannot be eliminated with the
normal pure Python file-read API; re-verify just before loading.

H199 official `ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()` stays EMPTY.
All 96 H212 images remain **research-only**; no official TRAIN
admission, independently sealed DEV, pristine OOD, historical
source/fork and old RGBA completeness, Qwen2B weight provenance,
LoRA optimizer steps or model accuracy measurements are claimed.

When reviewing H218, examine exact head Windows source SHA receipt
and hosted Linux stdlib-only safety tests; H218 CI is NOT a real
processor/model run. Keep all PRs Draft and protected main unchanged.
