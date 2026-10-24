# H102 — Original TRAIN geometry and EVAL frame audit

H100 and H101 both scored 0/16 strict GUI successes per arm. H101 improved
native JSON but generated even more degenerate drag outputs; **no evidence of
visual grounding improvement**. H102 is an independent **data/coordinate
pipeline** audit, not another model or training run.

Two strictly separate qualifications:

1. **Original public EVAL**: owner-only SHA-256 checks of source/framed images,
   independent recomputation of source→1024×640 frame dimensions, exact bbox
   scaling and H61 prompt. Ground-truth box-center actions must score 100%
   with the **unchanged H61 scorer**. This only qualifies evaluation geometry,
   never model performance or source-free test results.
2. **Original private TRAIN**: free CPU Kaggle only. SHA-admit the original
   1.76 GB H71 owner-private archive, independently rederive the immutable
   10,000-row H61 curriculum, inspect EVERY original label's bounds,
   coordinate-in-box, prompt/source/ordered identity and degeneracy.
   For the **actual H99 128 TRAIN rows**, SHA-check each original JPEG/PNG and
   actual decoded dimensions against the original label, strict action syntax,
   and original H61 geometry scorer. Emit aggregate counts only.
   **Do not rebalance, mutate, train or publish any TRAIN records.**

The source TRAIN is attached as a private Kaggle dataset, not downloaded
to Windows or committed to GitHub. No GPU required; original 10k/128 source
and selected audit quantities must be independently verifiable.

**H102 does not claim Qwen image processor tensors/tokenization are audited.**
The remaining hard gate is real processor image-token alignment and supervision
mask audit, then subsequent 2B supervised grounding experiments. Any H102
oracle success is *label/geometry correctness*, not model intelligence.
No private model publication, deployment or merges.
