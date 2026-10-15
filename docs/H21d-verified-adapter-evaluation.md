# H21d verified adapter evaluation

Implemented with CPU fake-boundary evidence; real adapter loading, fresh-session
retrieval and quality comparison remain unqualified. The operator must first
archive the trained asset using H21a and obtain its exact external manifest hash
and five identity fields from the signed archive source. A pretraining roundtrip
is never accepted as an adapter.

The existing `evaluation.development_runtime` entry point accepts these additional
flags together:

```text
--adapter /content/private-operator-download/adapter
--adapter-manifest-sha256 <64 lowercase hex>
--adapter-base-model Qwen/Qwen3-VL-2B-Instruct
--adapter-base-revision 89644892e4d85e24eaac8bacfd4f463576704203
--adapter-source-commit <40 lowercase hex producer source>
--adapter-data-sha256 <64 lowercase hex producer training data>
--adapter-evaluation-sha256 <64 lowercase hex producer baseline>
```

The normal bundle, font, frozen manifest, published runtime source and output
arguments are still required; the split still defaults to dev. The archived producer source commit is
separate from the current published evaluation source commit. Adapter identities
must come from the external archival record, never inferred from downloaded
bytes. Missing identity flags and unexpected flags fail closed. Retrieval is an
operator action in a fresh approved remote session; this runtime adds no URL
fetching or credentials. No adapter asset is included in this source change.

Before any ML imports or model allocation, the archive module validates its
bounded exact file inventory, manifest digest, identity, configuration and
safetensors header. Evaluation copies the original bytes into a private temporary
directory and revalidates that copy against the external pins. Configuration
bytes are never rewritten, including the original producer's null revision.
The exact base model revision is supplied explicitly to the base loader.

After the Colab and exact package gates, safetensors 0.6.2 decodes on CPU and
checks the actual complete 112-tensor inventory, shapes, supported floating
dtypes and every tensor's finite values. The unchanged FP16 base loads using the
existing settings and prompt path. PEFT 0.20.0 attaches the verified local
directory with `is_trainable=False` and `local_files_only=True`. Evaluation then
compares actual loaded adapter inventory, shapes, finite values and tensor
values against the decoded archive, and requires every model parameter frozen.
Missing weights, additional initialized modules and load errors fail without a
baseline fallback. The adapter path calls `eval()` and shares the baseline's
inference callback, generation settings, cases and independent scoring.

Baseline results retain `adapter_loaded=false`, `training_runs=0` and the existing
baseline schema. Candidate results use `hala.development.adapter_evaluation.v1`
and include the exact adapter manifest digest, identity and individual file
hashes/byte counts. `adapter_loaded` becomes true only after verified attachment;
the candidate process still reports `training_runs=0`. Failed attachment remains
a blocked result with `adapter_loaded=false`. Neither result grants execution
authority, native qualification or quality qualification.

CPU tests cover rejected identity/CLI boundaries before ML imports, exact private
config bytes, missing and extra files, roundtrip rejection, every finite tensor,
unexpected loaded modules, changed weights, trainable parameters, no load-error
fallback and the full fake runtime's baseline/candidate task and load order.
These tests do not establish real GPU behavior or quality improvement.

The full CPU suite passed 206 tests with one skip in 30.661 seconds on Windows
using the bundled Python runtime. No ML package installation or model download
was used. `git diff --check` passed. Fresh independent combined/security reviews found no must-fix; eight new tests independently passed. Genuine Jev judgments and Graft graph qualification are recorded in docs/reviews/h21d-source-gates.json. These qualify source only.

Remaining gates: signed DCO source publication; then a real unchanged-base run,
bounded training, trained archive publication, fresh-session operator retrieval,
verified adapter loading and comparison on the same frozen independent cases.
